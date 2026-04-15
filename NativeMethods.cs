using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace DekapuAutoOpencv;

internal static class NativeMethods
{
    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    private const int WH_KEYBOARD_LL = 13;

    private const int WM_KEYDOWN = 0x0100;

    public const ushort VK_TAB = 0x09;
    public const ushort VK_F5 = 0x74;
    public const ushort VK_F6 = 0x75;
    public const ushort VK_ESCAPE = 0x1B;

    private const uint INPUT_MOUSE = 0;
    private const uint INPUT_KEYBOARD = 1;

    private const uint MOUSEEVENTF_MOVE = 0x0001;
    private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
    private const uint MOUSEEVENTF_LEFTUP = 0x0004;
    private const uint MOUSEEVENTF_ABSOLUTE = 0x8000;
    private const uint MOUSEEVENTF_VIRTUALDESK = 0x4000;

    private const uint KEYEVENTF_KEYUP = 0x0002;

    private const int SM_CXVIRTUALSCREEN = 78;
    private const int SM_CYVIRTUALSCREEN = 79;
    private const int SM_XVIRTUALSCREEN = 76;
    private const int SM_YVIRTUALSCREEN = 77;

    // -------------------------------------------------------------------------
    // Structs
    // -------------------------------------------------------------------------

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT
    {
        public int Left, Top, Right, Bottom;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct POINT
    {
        public int X, Y;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MOUSEINPUT
    {
        public int dx, dy;
        public uint mouseData, dwFlags, time;
        public IntPtr dwExtraInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct KEYBDINPUT
    {
        public ushort wVk, wScan;
        public uint dwFlags, time;
        public IntPtr dwExtraInfo;
    }

    [StructLayout(LayoutKind.Explicit)]
    private struct INPUT
    {
        [FieldOffset(0)] public uint type;
        [FieldOffset(8)] public MOUSEINPUT mi;
        [FieldOffset(8)] public KEYBDINPUT ki;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct KBDLLHOOKSTRUCT
    {
        public uint vkCode, scanCode, flags, time;
        public IntPtr dwExtraInfo;
    }

    // -------------------------------------------------------------------------
    // Delegates — stored as static fields to prevent GC collection
    // -------------------------------------------------------------------------

    private delegate IntPtr LowLevelKeyboardProc(int nCode, IntPtr wParam, IntPtr lParam);
    private static LowLevelKeyboardProc? _hookProc;

    // -------------------------------------------------------------------------
    // P/Invoke — Window management
    // -------------------------------------------------------------------------

    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

    [DllImport("user32.dll")]
    public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);

    [DllImport("user32.dll")]
    public static extern int GetWindowTextLength(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    // -------------------------------------------------------------------------
    // P/Invoke — Input
    // -------------------------------------------------------------------------

    [DllImport("user32.dll", SetLastError = true)]
    private static extern uint SendInput(uint nInputs, INPUT[] pInputs, int cbSize);

    [DllImport("user32.dll")]
    private static extern bool GetCursorPos(out POINT lpPoint);

    [DllImport("user32.dll")]
    private static extern int GetSystemMetrics(int nIndex);

    // -------------------------------------------------------------------------
    // P/Invoke — Hook
    // -------------------------------------------------------------------------

    [DllImport("user32.dll", SetLastError = true)]
    private static extern IntPtr SetWindowsHookEx(int idHook, LowLevelKeyboardProc lpfn, IntPtr hMod, uint dwThreadId);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool UnhookWindowsHookEx(IntPtr hhk);

    [DllImport("user32.dll")]
    private static extern IntPtr CallNextHookEx(IntPtr hhk, int nCode, IntPtr wParam, IntPtr lParam);

    [DllImport("kernel32.dll", CharSet = CharSet.Auto)]
    private static extern IntPtr GetModuleHandle(string? lpModuleName);

    // -------------------------------------------------------------------------
    // P/Invoke — DPI
    // -------------------------------------------------------------------------

    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();

    // -------------------------------------------------------------------------
    // Public helpers
    // -------------------------------------------------------------------------

    public static (int X, int Y) GetCursorPosition()
    {
        GetCursorPos(out var p);
        return (p.X, p.Y);
    }

    public static string GetWindowTitle(IntPtr hwnd)
    {
        int len = GetWindowTextLength(hwnd);
        if (len == 0) return string.Empty;
        var sb = new StringBuilder(len + 1);
        GetWindowText(hwnd, sb, sb.Capacity);
        return sb.ToString();
    }

    public static void SendMouseMove(int screenX, int screenY)
    {
        int vsWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        int vsHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);
        int vsLeft = GetSystemMetrics(SM_XVIRTUALSCREEN);
        int vsTop = GetSystemMetrics(SM_YVIRTUALSCREEN);

        int normX = (int)((screenX - vsLeft) * 65535.0 / vsWidth);
        int normY = (int)((screenY - vsTop) * 65535.0 / vsHeight);

        var input = new INPUT { type = INPUT_MOUSE };
        input.mi.dx = normX;
        input.mi.dy = normY;
        input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK;
        SendInput(1, [input], Marshal.SizeOf<INPUT>());
    }

    public static void SendMouseButton(bool isDown)
    {
        var input = new INPUT { type = INPUT_MOUSE };
        input.mi.dwFlags = isDown ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_LEFTUP;
        SendInput(1, [input], Marshal.SizeOf<INPUT>());
    }

    public static void SendKey(ushort vk, bool keyUp)
    {
        var input = new INPUT { type = INPUT_KEYBOARD };
        input.ki.wVk = vk;
        input.ki.dwFlags = keyUp ? KEYEVENTF_KEYUP : 0u;
        SendInput(1, [input], Marshal.SizeOf<INPUT>());
    }

    public static IntPtr InstallKeyboardHook(Action onF5, Action onF6, Action onEsc)
    {
        _hookProc = (nCode, wParam, lParam) =>
        {
            if (nCode >= 0 && wParam == WM_KEYDOWN)
            {
                var kb = Marshal.PtrToStructure<KBDLLHOOKSTRUCT>(lParam);
                switch (kb.vkCode)
                {
                    case VK_F5: onF5(); break;
                    case VK_F6: onF6(); break;
                    case VK_ESCAPE: onEsc(); break;
                }
            }
            return CallNextHookEx(IntPtr.Zero, nCode, wParam, lParam);
        };

        IntPtr hMod;
        using (var process = Process.GetCurrentProcess())
        using (var module = process.MainModule!)
        {
            hMod = GetModuleHandle(module.ModuleName);
        }

        var hook = SetWindowsHookEx(WH_KEYBOARD_LL, _hookProc, hMod, 0);
        if (hook == IntPtr.Zero)
            throw new InvalidOperationException($"SetWindowsHookEx failed: {Marshal.GetLastWin32Error()}");

        return hook;
    }

    public static void UninstallKeyboardHook(IntPtr hook)
    {
        UnhookWindowsHookEx(hook);
        _hookProc = null;
    }
}
