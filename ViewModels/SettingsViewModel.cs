using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using DekapuAutoOpencv.Models;
using DekapuAutoOpencv.Services;

namespace DekapuAutoOpencv.ViewModels;

public partial class SettingsViewModel : ObservableObject
{
    private readonly AppSettings _settings;
    private readonly SettingsService _service;
    private readonly Action<(int W, int H)> _onSaved;

    // ── 動作設定 ──────────────────────────────────────
    [ObservableProperty]
    public partial int WindowWidth { get; set; }
    [ObservableProperty]
    public partial int WindowHeight { get; set; }
    [ObservableProperty]
    public partial int LoopWaitMs { get; set; }
    [ObservableProperty]
    public partial int DiffCaptureWaitMs { get; set; }
    [ObservableProperty]
    public partial bool DryRun { get; set; }

    // ── 検知設定 ──────────────────────────────────────
    [ObservableProperty]
    public partial int ThresholdValue { get; set; }
    [ObservableProperty]
    public partial int MinContourArea { get; set; }
    [ObservableProperty]
    public partial int DilateIterations { get; set; }

    // ── クリック設定 ──────────────────────────────────
    [ObservableProperty]
    public partial int MinClickDistance { get; set; }
    [ObservableProperty]
    public partial int MaxClicksPerLoop { get; set; }
    [ObservableProperty]
    public partial int RandomClicksPerContour { get; set; }
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsMinClickDistanceEnabled))]
    public partial bool EnableAvoidCloseClick { get; set; }

    public bool IsMinClickDistanceEnabled => EnableAvoidCloseClick;

    // ── 入力設定 ──────────────────────────────────────

    [ObservableProperty]
    public partial int MoveAfterWaitMs { get; set; }
    [ObservableProperty]
    public partial int ClickDownWaitMs { get; set; }
    [ObservableProperty]
    public partial double MouseMoveDivisor { get; set; }
    [ObservableProperty]
    public partial double FovSuppressionFactor { get; set; }

    public event Action? RequestClose;

    public SettingsViewModel(AppSettings settings, SettingsService service,
                             Action<(int W, int H)> onSaved)
    {
        _settings = settings;
        _service = service;
        _onSaved = onSaved;

        // 現在の設定値を各プロパティにコピー
        WindowWidth = settings.WindowWidth;
        WindowHeight = settings.WindowHeight;
        LoopWaitMs = settings.LoopWaitMs;
        DiffCaptureWaitMs = settings.DiffCaptureWaitMs;
        DryRun = settings.DryRun;

        ThresholdValue = settings.ThresholdValue;
        MinContourArea = settings.MinContourArea;
        DilateIterations = settings.DilateIterations;

        MinClickDistance = settings.MinClickDistance;
        MaxClicksPerLoop = settings.MaxClicksPerLoop;
        RandomClicksPerContour = settings.RandomClicksPerContour;
        EnableAvoidCloseClick = settings.EnableAvoidCloseClick;

        MoveAfterWaitMs = settings.MoveAfterWaitMs;
        ClickDownWaitMs = settings.ClickDownWaitMs;
        MouseMoveDivisor = settings.MouseMoveDivisor;
        FovSuppressionFactor = settings.FovSuppressionFactor;
    }

    [RelayCommand]
    private void Save()
    {
        var prevSize = (_settings.WindowWidth, _settings.WindowHeight);

        // 編集値を AppSettings に書き戻す
        _settings.WindowWidth = WindowWidth;
        _settings.WindowHeight = WindowHeight;
        _settings.LoopWaitMs = LoopWaitMs;
        _settings.DiffCaptureWaitMs = DiffCaptureWaitMs;
        _settings.DryRun = DryRun;

        _settings.ThresholdValue = ThresholdValue;
        _settings.MinContourArea = MinContourArea;
        _settings.DilateIterations = DilateIterations;

        _settings.MinClickDistance = MinClickDistance;
        _settings.MaxClicksPerLoop = MaxClicksPerLoop;
        _settings.RandomClicksPerContour = RandomClicksPerContour;
        _settings.EnableAvoidCloseClick = EnableAvoidCloseClick;

        _settings.MoveAfterWaitMs = MoveAfterWaitMs;
        _settings.ClickDownWaitMs = ClickDownWaitMs;
        _settings.MouseMoveDivisor = MouseMoveDivisor;
        _settings.FovSuppressionFactor = FovSuppressionFactor;

        _service.Save(_settings);
        _onSaved(prevSize);
        RequestClose?.Invoke();
    }

    [RelayCommand]
    private void Cancel() => RequestClose?.Invoke();
}
