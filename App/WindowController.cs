namespace DekapuAutoOpencv.Core;

public class WindowController((int Width, int Height) targetSize)
{
    private IntPtr _hwnd = IntPtr.Zero;
    private (int Width, int Height)? _originalSize;
    private (int Width, int Height) _targetSize = targetSize;

    public void UpdateTargetSize((int Width, int Height) newSize) => _targetSize = newSize;

    public IntPtr Hwnd => _hwnd;

    public (int Left, int Top, int Width, int Height)? Region
    {
        get
        {
            if (_hwnd == IntPtr.Zero) return null;
            NativeMethods.GetWindowRect(_hwnd, out var r);
            return (r.Left, r.Top, r.Right - r.Left, r.Bottom - r.Top);
        }
    }

    public bool SetWindow()
    {
        Restore();

        var hwnd = NativeMethods.GetForegroundWindow();
        if (hwnd == IntPtr.Zero)
        {
            Console.WriteLine("Active window not found.");
            return false;
        }

        NativeMethods.GetWindowRect(hwnd, out var r);
        _hwnd = hwnd;
        _originalSize = (r.Right - r.Left, r.Bottom - r.Top);

        Console.WriteLine($"Window set: {NativeMethods.GetWindowTitle(hwnd)}");
        return true;
    }

    public bool Resize()
    {
        if (_hwnd == IntPtr.Zero) return false;

        NativeMethods.GetWindowRect(_hwnd, out var r);
        bool ok = NativeMethods.MoveWindow(_hwnd, r.Left, r.Top,
            _targetSize.Width, _targetSize.Height, true);

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
