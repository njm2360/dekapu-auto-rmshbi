using System.Windows.Media;
using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using DekapuAutoOpencv.Core;
using DekapuAutoOpencv.Models;
using DekapuAutoOpencv.Services;
using OpenCvSharp;

namespace DekapuAutoOpencv.ViewModels;

public partial class MainViewModel : ObservableObject, IDisposable
{
    // -------------------------------------------------------------------------
    // Services & settings
    // -------------------------------------------------------------------------

    public readonly SettingsService SettingsService = new();
    public readonly AppSettings Settings;

    // -------------------------------------------------------------------------
    // Automation components
    // -------------------------------------------------------------------------

    private readonly WindowController _windowCtrl;
    private readonly InputController _inputCtrl;
    private readonly ImageCapture _capture;
    private readonly ClickPointExtractor _extractor;
    private MotionDetector _detector;
    private Mat _mask;

    // -------------------------------------------------------------------------
    // Async loop state
    // -------------------------------------------------------------------------

    private readonly Dispatcher _dispatcher;
    private CancellationTokenSource _cts = new();
    private volatile bool _running = false;

    // -------------------------------------------------------------------------
    // Observable properties
    // -------------------------------------------------------------------------

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(StatusText))]
    [NotifyPropertyChangedFor(nameof(StatusColor))]
    public partial bool IsRunning { get; set; } = false;

    [ObservableProperty]
    public partial string WindowTitle { get; set; } = "(未設定)";

    [ObservableProperty]
    public partial string LogText { get; set; } = string.Empty;

    public string StatusText => IsRunning ? "実行中" : "停止中";
    public Brush StatusColor => IsRunning ? Brushes.LimeGreen : Brushes.Gray;

    public MainViewModel()
    {
        _dispatcher = Dispatcher.CurrentDispatcher;

        Settings = SettingsService.Load();

        _windowCtrl = new WindowController((Settings.WindowWidth, Settings.WindowHeight));
        _inputCtrl = new InputController(_windowCtrl, Settings);
        _capture = new ImageCapture(_windowCtrl);
        _extractor = new ClickPointExtractor(Settings);
        _mask = MaskLoader.MakeDefault((Settings.WindowWidth, Settings.WindowHeight));
        _detector = new MotionDetector(Settings, _mask);

        _ = Task.Run(() => MainLoopAsync(_cts.Token));
    }

    public void OnF5()
    {
        if (_running)
        {
            Log("実行中はウィンドウを変更できません");
            return;
        }

        if (_windowCtrl.SetWindow())
        {
            _windowCtrl.Resize();
            WindowTitle = NativeMethods.GetWindowTitle(_windowCtrl.Hwnd);
            Log($"ウィンドウを設定: {WindowTitle}");
        }
        else
        {
            Log("ウィンドウの設定に失敗しました");
        }
    }

    public void OnF6()
    {
        if (_running) return;

        if (_windowCtrl.Hwnd == IntPtr.Zero)
        {
            Log("ウィンドウが設定されていません");
            return;
        }

        _running = true;
        IsRunning = true;
        Log("開始しました");

        _ = Task.Run(() => _inputCtrl.PerspectiveLockAsync());
    }

    public void OnEsc()
    {
        if (!_running) return;

        _running = false;
        IsRunning = false;
        Log("停止しました");
    }

    public void OnSettingsSaved((int W, int H) prevSize)
    {
        var newSize = (Settings.WindowWidth, Settings.WindowHeight);
        if (newSize != prevSize)
        {
            _mask.Dispose();
            _mask = MaskLoader.MakeDefault(newSize);
            _detector = new MotionDetector(Settings, _mask);
            _windowCtrl.UpdateTargetSize(newSize);
            Log($"ウィンドウサイズを {newSize.Item1}×{newSize.Item2} に変更しました");
        }
        else
        {
            Log("設定を保存しました");
        }
    }

    // -------------------------------------------------------------------------
    // Main automation loop
    // -------------------------------------------------------------------------

    private async Task MainLoopAsync(CancellationToken ct)
    {
        try
        {
            while (!ct.IsCancellationRequested)
            {
                if (_running)
                {
                    await LoopStepAsync();
                    await Task.Delay(Settings.LoopWaitMs, ct);
                }
                else
                {
                    await Task.Delay(Settings.IdleWaitMs, ct);
                }
            }
        }
        catch (OperationCanceledException) { }
        finally
        {
            _inputCtrl.Cleanup();
            _windowCtrl.Restore();
        }
    }

    private async Task LoopStepAsync()
    {
        var (prev, curr) = await _capture.CapturePairAsync();
        if (prev is null || curr is null) return;

        try
        {
            var contours = _detector.Detect(prev, curr);
            if (contours.Length == 0) return;

            var clicks = _extractor.Extract(contours);
            if (clicks.Count == 0) return;

            await _inputCtrl.ExecuteClicksAsync(clicks, Settings.DryRun);
        }
        finally
        {
            prev.Dispose();
            curr.Dispose();
        }
    }

    // -------------------------------------------------------------------------
    // Logging (thread-safe)
    // -------------------------------------------------------------------------

    public void Log(string message)
    {
        var line = $"[{DateTime.Now:HH:mm:ss}] {message}";

        _dispatcher.BeginInvoke(() =>
        {
            const int MaxLines = 200;
            var lines = LogText.Length == 0
                ? []
                : LogText.Split('\n');

            var kept = lines.Length >= MaxLines
                ? lines[(lines.Length - MaxLines + 1)..]
                : lines;

            LogText = kept.Length == 0
                ? line
                : string.Join('\n', kept) + '\n' + line;
        });
    }

    // -------------------------------------------------------------------------
    // IDisposable
    // -------------------------------------------------------------------------

    public void Dispose()
    {
        _cts.Cancel();
        _mask.Dispose();
        _cts.Dispose();
    }
}
