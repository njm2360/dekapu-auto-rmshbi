using DekapuAutoOpencv.Models;

namespace DekapuAutoOpencv.Core;

public class InputController(WindowController windowController, AppSettings settings)
{
    private readonly SemaphoreSlim _lock = new(1, 1);
    private Point? _origin;

    public async Task ExecuteClicksAsync(IReadOnlyList<Point> points, bool dryRun = false, CancellationToken ct = default)
    {
        if (_origin is null)
            throw new InvalidOperationException("Origin is not set");
        if (points.Count == 0)
            return;

        var corrected = points.Select(Correct).ToList();

        await _lock.WaitAsync(ct);
        try
        {
            foreach (var point in corrected)
            {
                ct.ThrowIfCancellationRequested();

                MoveMouseTo(point);

                if (dryRun)
                {
                    await Task.Delay(1000, ct);
                }
                else
                {
                    await Task.Delay(settings.MoveAfterWaitMs, ct);
                    await ClickAsync(ct);
                }
            }

            MoveMouseTo(_origin.Value);
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task PerspectiveLockAsync()
    {
        var (cx, cy) = windowController.ClientCenter
            ?? throw new InvalidOperationException("Window is not set");

        NativeMethods.SendKey(NativeMethods.VK_TAB, keyUp: false);

        _origin = new Point(cx, cy);
    }

    public static void Cleanup()
    {
        NativeMethods.SendMouseButton(isDown: false);
        NativeMethods.SendKey(NativeMethods.VK_TAB, keyUp: true);
    }

    public Point Correct(Point point)
    {
        var region = windowController.Region
            ?? throw new InvalidOperationException("Window is not set");
        if (_origin is null)
            throw new InvalidOperationException("Origin is not set");

        var (wLeft, wTop, wWidth, _) = region;

        var absPoint = new Point(wLeft, wTop) + point;
        double halfWidth = wWidth / 2.0;
        double distX = Math.Abs(point.X - halfWidth);
        double normX = distX / halfWidth;
        double suppression = Math.Max(0.0, 1.0 - (normX * settings.FovSuppressionFactor));

        var delta = absPoint - _origin.Value;

        return _origin.Value + new Point(
            (int)(delta.X * suppression / settings.MouseMoveDivisor),
            (int)(delta.Y / settings.MouseMoveDivisor));
    }

    private async Task ClickAsync(CancellationToken ct)
    {
        NativeMethods.SendMouseButton(isDown: true);
        try
        {
            await Task.Delay(settings.ClickDownWaitMs, ct);
        }
        finally
        {
            NativeMethods.SendMouseButton(isDown: false);
        }
    }

    private static void MoveMouseTo(Point p) =>
        NativeMethods.SendMouseMove(p.X, p.Y);
}
