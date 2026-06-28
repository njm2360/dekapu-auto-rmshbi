namespace DekapuAutoOpencv.Core;

public class WindowController((int Width, int Height) targetSize)
{
    private IntPtr _hwnd = IntPtr.Zero;
    private (int Width, int Height)? _originalSize;
    private (int Width, int Height) _targetSize = targetSize;

    public void UpdateTargetSize((int Width, int Height) newSize) => _targetSize = newSize;

    public IntPtr Hwnd => _hwnd;

    public bool IsForeground =>
        _hwnd != IntPtr.Zero && NativeMethods.GetForegroundWindow() == _hwnd;

    public (int Left, int Top, int Width, int Height)? Region
    {
        get
        {
            if (_hwnd == IntPtr.Zero) return null;
            var r = NativeMethods.GetVisibleRect(_hwnd);
            return (r.Left, r.Top, r.Right - r.Left, r.Bottom - r.Top);
        }
    }

    public (int X, int Y)? ClientCenter =>
        _hwnd == IntPtr.Zero ? null : NativeMethods.GetClientCenter(_hwnd);

    public bool SetWindow()
    {
        var hwnd = NativeMethods.GetForegroundWindow();
        if (hwnd == IntPtr.Zero)
        {
            Console.WriteLine("Active window not found.");
            return false;
        }

        if (NativeMethods.IsWindowOwnedByCurrentProcess(hwnd))
        {
            Console.WriteLine("Cannot set own window as target.");
            return false;
        }

        if (hwnd == _hwnd)
            return true;

        Restore();

        NativeMethods.GetWindowRect(hwnd, out var r);
        _hwnd = hwnd;
        _originalSize = (r.Right - r.Left, r.Bottom - r.Top);

        Console.WriteLine($"Window set: {NativeMethods.GetWindowTitle(hwnd)}");
        return true;
    }

    public bool Resize()
    {
        if (_hwnd == IntPtr.Zero) return false;

        NativeMethods.GetWindowRect(_hwnd, out var outer);
        var visible = NativeMethods.GetVisibleRect(_hwnd);
        int borderX = outer.Right - outer.Left - (visible.Right - visible.Left);
        int borderY = outer.Bottom - outer.Top - (visible.Bottom - visible.Top);

        bool ok = NativeMethods.MoveWindow(_hwnd, outer.Left, outer.Top,
            _targetSize.Width + borderX, _targetSize.Height + borderY, true);

        if (!ok)
            Console.WriteLine("Resize failed.");

        return ok;
    }

    public void Restore()
    {
        if (_hwnd == IntPtr.Zero || _originalSize is null) return;

        NativeMethods.GetWindowRect(_hwnd, out var r);
        NativeMethods.MoveWindow(_hwnd, r.Left, r.Top,
            _originalSize.Value.Width, _originalSize.Value.Height, true);

        Console.WriteLine("Window size restored.");
        _hwnd = IntPtr.Zero;
        _originalSize = null;
    }
}
