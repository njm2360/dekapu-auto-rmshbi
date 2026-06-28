using DekapuAutoOpencv.Models;
using OpenCvSharp;

using OcvPoint = OpenCvSharp.Point;

namespace DekapuAutoOpencv.Core;

public class MotionDetector(AppSettings settings, Mat? mask = null)
{
    public OcvPoint[][] Detect(Mat prevImg, Mat currImg)
    {
        using var diff = new Mat();
        using var gray = new Mat();
        using var thresh = new Mat();
        using var dilated = new Mat();

        Cv2.Absdiff(prevImg, currImg, diff);
        Cv2.CvtColor(diff, gray, ColorConversionCodes.BGR2GRAY);
        Cv2.Threshold(gray, thresh, settings.ThresholdValue, 255, ThresholdTypes.Binary);
        Cv2.Dilate(thresh, dilated, null, iterations: settings.DilateIterations);

        if (mask != null)
        {
            if (mask.Size() != dilated.Size())
                Cv2.Resize(mask, mask, dilated.Size(), 0, 0, InterpolationFlags.Nearest);

            Cv2.BitwiseAnd(dilated, mask, dilated);
        }

        Cv2.FindContours(
            dilated,
            out var contours,
            out _,
            RetrievalModes.External,
            ContourApproximationModes.ApproxSimple);

        if (contours.Length == 0)
            return [];

        var filtered = contours
            .Where(c => Cv2.ContourArea(c) >= settings.MinContourArea)
            .OrderByDescending(c => Cv2.ContourArea(c))
            .ToArray();

        return filtered;
    }

    public static Mat Annotate(OcvPoint[][] contours, Mat image)
    {
        var output = image.Clone();

        foreach (var c in contours)
        {
            var rect = Cv2.BoundingRect(c);
            var area = (int)Cv2.ContourArea(c);

            Cv2.DrawContours(output, [c], -1, new Scalar(0, 255, 0), 2);

            const double fontSize = 0.55;
            const int fontThickness = 1;
            var label = $"{area}px ({rect.Width}x{rect.Height})";
            var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, fontSize, fontThickness, out var baseline);

            var textOrg = new OcvPoint(rect.X, rect.Y > textSize.Height + 4 ? rect.Y - 4 : rect.Y + textSize.Height + 4);
            var bgTl = new OcvPoint(textOrg.X - 1, textOrg.Y - textSize.Height - 1);
            var bgBr = new OcvPoint(textOrg.X + textSize.Width + 1, textOrg.Y + baseline + 1);

            Cv2.Rectangle(output, bgTl, bgBr, new Scalar(255, 255, 255), -1);
            Cv2.PutText(output, label, textOrg, HersheyFonts.HersheySimplex, fontSize, new Scalar(0, 0, 0), fontThickness, LineTypes.AntiAlias);
        }

        return output;
    }
}
