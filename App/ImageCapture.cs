using System.Drawing;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace DekapuAutoOpencv.Core;

public class ImageCapture(WindowController windowController)
{
    public async Task<(Mat? Prev, Mat? Curr)> CapturePairAsync(double delay = 0.1)
    {
        var region = windowController.Region;
        if (region is null)
        {
            Console.WriteLine("Capture failed: window not set.");
            return (null, null);
        }

        try
        {
            var img1 = await Task.Run(() => Shot(region.Value));
            await Task.Delay(TimeSpan.FromSeconds(delay));
            var img2 = await Task.Run(() => Shot(region.Value));
            return (img1, img2);
        }
        catch (Exception e)
        {
            Console.WriteLine($"Screenshot failed: {e.Message}");
            return (null, null);
        }
    }

    private static Mat Shot((int Left, int Top, int Width, int Height) region)
    {
        var (left, top, width, height) = region;

        using var bmp = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        using var g = Graphics.FromImage(bmp);
        g.CopyFromScreen(left, top, 0, 0, new System.Drawing.Size(width, height));

        // BitmapConverter.ToMat returns CV_8UC4 (BGRA) for 32bpp bitmaps
        var mat = BitmapConverter.ToMat(bmp);

        // Convert BGRA → BGR to match Python's cv2.cvtColor(RGB → BGR) pipeline
        Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2BGR);
        return mat;
    }
}
