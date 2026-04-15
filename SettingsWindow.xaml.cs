using System.Windows;
using DekapuAutoOpencv.ViewModels;

namespace DekapuAutoOpencv;

public partial class SettingsWindow : Window
{
    public SettingsWindow()
    {
        InitializeComponent();
    }

    protected override void OnContentRendered(EventArgs e)
    {
        base.OnContentRendered(e);
        if (DataContext is SettingsViewModel vm)
            vm.RequestClose += Close;
    }

    protected override void OnClosed(EventArgs e)
    {
        if (DataContext is SettingsViewModel vm)
            vm.RequestClose -= Close;
        base.OnClosed(e);
    }
}
