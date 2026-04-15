namespace DekapuAutoOpencv.Core;

public readonly record struct Point(int X, int Y)
{
    public static Point operator +(Point a, Point b) => new(a.X + b.X, a.Y + b.Y);
    public static Point operator -(Point a, Point b) => new(a.X - b.X, a.Y - b.Y);
    public static Point operator *(Point p, double factor) => new((int)(p.X * factor), (int)(p.Y * factor));

    public Point Offset(int dx = 0, int dy = 0) => new(X + dx, Y + dy);

    public double DistanceTo(Point other) =>
        Math.Sqrt(Math.Pow(X - other.X, 2) + Math.Pow(Y - other.Y, 2));
}
