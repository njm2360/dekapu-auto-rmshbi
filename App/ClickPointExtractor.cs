using DekapuAutoOpencv.Models;
using OpenCvSharp;

using OcvPoint = OpenCvSharp.Point;

namespace DekapuAutoOpencv.Core;

public enum ContourClickMode
{
    Center = 0,
    Random = 1,
}

public class ClickPointExtractor(AppSettings settings, ContourClickMode mode = ContourClickMode.Random)
{
    private readonly Random _rng = new();

    public List<Point> Extract(OcvPoint[][] contours)
    {
        var points = new List<Point>();

        foreach (var contour in contours)
        {
            if (points.Count >= settings.MaxClicksPerLoop) break;

            IEnumerable<Point?> candidates = mode switch
            {
                ContourClickMode.Random => Enumerable
                    .Range(0, settings.RandomClicksPerContour)
                    .Select(_ => RandomPoint(contour)),
                ContourClickMode.Center => [CenterPoint(contour)],
                _ => [],
            };

            foreach (var candidate in candidates)
            {
                if (candidate is null) continue;
                if (settings.EnableAvoidCloseClick && !FarEnough(candidate.Value, points)) continue;
                points.Add(candidate.Value);
            }
        }

        return points;
    }

    private Point? RandomPoint(OcvPoint[] contour)
    {
        var rect = Cv2.BoundingRect(contour);

        for (int i = 0; i < 5; i++)
        {
            // Python random.randint(a, b) is inclusive on both ends
            // C# Random.Next(a, b) is exclusive on upper end → use +1
            int x = _rng.Next(rect.X, rect.X + rect.Width + 1);
            int y = _rng.Next(rect.Y, rect.Y + rect.Height + 1);

            if (Cv2.PointPolygonTest(contour, new Point2f(x, y), false) >= 0)
                return new Point(x, y);
        }

        return null;
    }

    private static Point? CenterPoint(OcvPoint[] contour)
    {
        var m = Cv2.Moments(contour);
        if (m.M00 == 0) return null;
        return new Point((int)(m.M10 / m.M00), (int)(m.M01 / m.M00));
    }

    private bool FarEnough(Point p, List<Point> existing) =>
        existing.All(e => p.DistanceTo(e) >= settings.MinClickDistance);
}
