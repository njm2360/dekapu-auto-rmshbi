using System.ComponentModel;
using System.Windows;
using DekapuAutoOpencv.ViewModels;

namespace DekapuAutoOpencv;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    protected override void OnContentRendered(EventArgs e)
    {
        base.OnContentRendered(e);

        if (DataContext is MainViewModel vm)
            vm.PropertyChanged += OnViewModelPropertyChanged;
    }

    protected override void OnClosed(EventArgs e)
    {
        if (DataContext is MainViewModel vm)
            vm.PropertyChanged -= OnViewModelPropertyChanged;

        base.OnClosed(e);
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(MainViewModel.LogText))
            LogScroller.ScrollToEnd();
    }

    private void OpenSettings_Click(object sender, RoutedEventArgs e)
    {
        var mainVm = (MainViewModel)DataContext;
        var settingsVm = new SettingsViewModel(
            mainVm.Settings,
            mainVm.SettingsService,
            prevSize => mainVm.OnSettingsSaved(prevSize));

        new SettingsWindow { DataContext = settingsVm, Owner = this }.ShowDialog();
    }
}
