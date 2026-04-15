using System.IO;
using OpenCvSharp;

namespace DekapuAutoOpencv.Core;

public static class MaskLoader
{
    public static Mat MakeDefault((int Width, int Height) size)
    {
        var (w, h) = size;
        var mask = Mat.Zeros(h, w, MatType.CV_8UC1).ToMat();

        using var lowerHalf = new Mat(h - h / 2, w, MatType.CV_8UC1, new Scalar(255));
        lowerHalf.CopyTo(mask[new Rect(0, h / 2, w, h - h / 2)]);

        return mask;
    }

    public static Mat Load(string maskPath, (int Width, int Height) size)
    {
        if (!File.Exists(maskPath))
        {
            Console.WriteLine("Mask file not found. Using default mask");
            return MakeDefault(size);
        }

        var maskGray = Cv2.ImRead(maskPath, ImreadModes.Grayscale);
        if (maskGray.Empty())
        {
            Console.WriteLine("Failed to load mask image. Using default mask");
            return MakeDefault(size);
        }

        if (maskGray.Width != size.Width || maskGray.Height != size.Height)
        {
            Console.WriteLine(
                $"Mask size {maskGray.Width}x{maskGray.Height} does not match " +
                $"WINDOW_SIZE {size}. Using default mask");
            maskGray.Dispose();
            return MakeDefault(size);
        }

        Cv2.Threshold(maskGray, maskGray, 1, 255, ThresholdTypes.Binary);
        Console.WriteLine("Mask image loaded successfully");
        return maskGray;
    }
}
