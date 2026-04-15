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
            Cv2.BitwiseAnd(dilated, mask, dilated);

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

    public void Debug(OcvPoint[][] contours, Mat image)
    {
        using var output = image.Clone();

        foreach (var c in contours)
        {
            double area = Cv2.ContourArea(c);
            Cv2.DrawContours(output, new[] { c }, -1, new Scalar(0, 255, 0), 2);

            var rect = Cv2.BoundingRect(c);
            var textPos = new OcvPoint(rect.X, rect.Y > 5 ? rect.Y - 5 : rect.Y + 15);

            Cv2.PutText(
                output,
                $"{(int)area}",
                textPos,
                HersheyFonts.HersheySimplex,
                0.6,
                new Scalar(0, 0, 255),
                2,
                LineTypes.AntiAlias);
        }

        Cv2.ImWrite("output.png", output);
    }
}
