namespace DekapuAutoOpencv.Models;

public class AppSettings
{
    // ── 動作設定 ──────────────────────────────────────
    public int WindowWidth { get; set; } = 1024;
    public int WindowHeight { get; set; } = 768;
    public int LoopWaitMs { get; set; } = 3000;
    public bool DryRun { get; set; } = false;

    // ── 検知設定 ──────────────────────────────────────
    public int ThresholdValue { get; set; } = 30;
    public int MinContourArea { get; set; } = 200;
    public int DilateIterations { get; set; } = 2;

    // ── クリック設定 ──────────────────────────────────
    public int MinClickDistance { get; set; } = 20;
    public int MaxClicksPerLoop { get; set; } = 10;
    public int RandomClicksPerContour { get; set; } = 3;
    public bool EnableAvoidCloseClick { get; set; } = true;

    // ── 入力設定 ──────────────────────────────────────
    public int MouseTakeWaitMs { get; set; } = 100;
    public int MoveAfterWaitMs { get; set; } = 100;
    public int ClickDownWaitMs { get; set; } = 100;
    public double MouseMoveDivisor { get; set; } = 2.0;
    public double FovSuppressionFactor { get; set; } = 0.2;
}
