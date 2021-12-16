using System;
using UnityEngine;

public interface ISquasher
{
    public Vector2 Squash(Vector2 input);
}

public class Squasher
{
    public enum SquashersEnum
    {
        Clamp,
        TanhClamp,
        RadialTanh,
        Identity
    }

    public static Func<Vector2, Vector2> GetSquasher(SquashersEnum squasher)
    {
        Func<Vector2, Vector2> squasherFunc = squasher switch
        {
            SquashersEnum.Clamp => Clamp,
            SquashersEnum.TanhClamp => TanhClamp,
            SquashersEnum.RadialTanh => RadialTanh,
            SquashersEnum.Identity => Identity,
            _ => null
        };
        return squasherFunc;
    }

    public static Vector2 Clamp(Vector2 vector)
    {
        var velocity = new Vector2(Mathf.Clamp(vector.x, -1f, 1f), Mathf.Clamp(vector.y, -1f, 1f));
        velocity = Vector2.ClampMagnitude(velocity, 1f);
        return velocity;
    }

    public static Vector2 TanhClamp(Vector2 vector)
    {
        var velocity = new Vector2(MathF.Tanh(vector.x), MathF.Tanh(vector.y));
        velocity = Vector2.ClampMagnitude(velocity, 1f);
        return velocity;
    }

    public static Vector2 RadialTanh(Vector2 vector)
    {
        return vector.normalized * MathF.Tanh(vector.magnitude);
    }

    public static Vector2 Identity(Vector2 vector)
    {
        return vector;
    }
    
}
