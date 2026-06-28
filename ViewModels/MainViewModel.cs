using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
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
    [NotifyPropertyChangedFor(nameof(ThumbnailPlaceholderVisibility))]
    public partial BitmapSource? ThumbnailImage { get; set; } = null;

    public string StatusText => IsRunning ? "実行中" : "停止中";
    public Brush StatusColor => IsRunning ? Brushes.LimeGreen : Brushes.Gray;
    public Visibility ThumbnailPlaceholderVisibility => ThumbnailImage is null ? Visibility.Visible : Visibility.Collapsed;

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

    public void OnSetWindow()
    {
        if (_running) return;

        if (_windowCtrl.SetWindow())
        {
            _windowCtrl.Resize();
            WindowTitle = NativeMethods.GetWindowTitle(_windowCtrl.Hwnd);
        }
    }

    public void OnStart()
    {
        if (_running) return;
        if (_windowCtrl.Hwnd == IntPtr.Zero) return;

        _running = true;
        IsRunning = true;

        _ = Task.Run(() => _inputCtrl.PerspectiveLockAsync());
    }

    public void OnStop()
    {
        if (!_running) return;

        _running = false;
        IsRunning = false;
        ThumbnailImage = null;

        _inputCtrl.Cleanup();
    }

    public void OnSettingsSaved((int W, int H) prevSize)
    {
        var newSize = (Settings.WindowWidth, Settings.WindowHeight);
        if (newSize == prevSize) return;

        _mask.Dispose();
        _mask = MaskLoader.MakeDefault(newSize);
        _detector = new MotionDetector(Settings, _mask);
        _windowCtrl.UpdateTargetSize(newSize);
    }

    // -------------------------------------------------------------------------
    // Main automation loop
    // -------------------------------------------------------------------------

    private async Task MainLoopAsync(CancellationToken ct)
    {
        using var timer = new PeriodicTimer(TimeSpan.FromMilliseconds(Settings.LoopWaitMs));
        try
        {
            while (await timer.WaitForNextTickAsync(ct))
            {
                if (!_running) continue;

                if (!_windowCtrl.IsForeground)
                {
                    _ = _dispatcher.BeginInvoke(OnStop);
                    continue;
                }

                await LoopStepAsync();
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

            UpdateThumbnail(curr, contours);

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

    private void UpdateThumbnail(Mat mat, OpenCvSharp.Point[][] contours)
    {
        const int DisplayWidth = 448;
        var scale = Math.Min(1.0, (double)DisplayWidth / mat.Width);
        var size = new OpenCvSharp.Size((int)(mat.Width * scale), (int)(mat.Height * scale));

        using var resized = mat.Resize(size, interpolation: InterpolationFlags.Area);

        var scaledContours = contours
            .Select(c => c.Select(p => new OpenCvSharp.Point((int)(p.X * scale), (int)(p.Y * scale))).ToArray())
            .ToArray();

        using var annotated = MotionDetector.Annotate(scaledContours, resized);
        Cv2.ImEncode(".png", annotated, out var buf);

        _dispatcher.BeginInvoke(() =>
        {
            using var ms = new MemoryStream(buf);
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.StreamSource = ms;
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.EndInit();
            bmp.Freeze();
            ThumbnailImage = bmp;
        });
    }

    // -------------------------------------------------------------------------
    // IDisposable
    // -------------------------------------------------------------------------

    public void Dispose()
    {
        _cts.Cancel();
        _inputCtrl.Cleanup();
        _windowCtrl.Restore();
        _mask.Dispose();
        _cts.Dispose();
    }
}
