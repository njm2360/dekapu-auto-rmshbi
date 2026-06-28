using System.Windows;
using DekapuAutoOpencv.ViewModels;

namespace DekapuAutoOpencv;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    private void OpenSettings_Click(object sender, RoutedEventArgs e)
    {
        var mainVm = (MainViewModel)DataContext;
        var settingsVm = new SettingsViewModel(
            mainVm.Settings,
            mainVm.SettingsService,
            mainVm.OnSettingsSaved);

        new SettingsWindow { DataContext = settingsVm, Owner = this }.ShowDialog();
    }
}
