using System.Windows;
using DekapuAutoOpencv.ViewModels;

namespace DekapuAutoOpencv;

public partial class App : Application
{
    public MainViewModel ViewModel { get; private set; } = null!;
    private IntPtr _hookHandle = IntPtr.Zero;

    private void OnStartup(object sender, StartupEventArgs e)
    {
        NativeMethods.SetProcessDPIAware();

        ViewModel = new MainViewModel();

        _hookHandle = NativeMethods.InstallKeyboardHook(
            onF5: ViewModel.OnF5,
            onF6: ViewModel.OnF6,
            onEsc: ViewModel.OnEsc);

        var window = new MainWindow { DataContext = ViewModel };
        MainWindow = window;
        window.Show();
    }

    private void OnExit(object sender, ExitEventArgs e)
    {
        if (_hookHandle != IntPtr.Zero)
            NativeMethods.UninstallKeyboardHook(_hookHandle);

        ViewModel?.Dispose();
    }
}
