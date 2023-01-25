using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using UnityEngine;
using Random = UnityEngine.Random;

public class MLUtils
{

    public static Vector3 NoncollidingPosition(
        float xMin,
        float xMax,
        float zMin,
        float zMax,
        float yVal,
        List<Vector3> excludes,
        int maxTries = 10,
        float threshold = 0.5f)
    {
        Vector3 position = new Vector3(
            Random.Range(xMin, xMax), 
            yVal,
            Random.Range(zMin, zMax)
        );

        var found = false;
        for (var i = 0; i < maxTries; i++)
        {

            var valid = excludes
                .Select(v => new Vector3(v.x, yVal, v.z))
                .All(p => (p - position).magnitude > threshold);
            
            // Debug.Log($"Comparing against {excludes.Count} agents");

            if (valid)
            {
                found = true;
                // Debug.Log($"Found a location after {i} tries");
                break;
            }
            
            position = new Vector3(
                Random.Range(xMin, xMax), 
                yVal,
                Random.Range(zMin, zMax)
            );
        }

        if (!found)
        {
            Debug.Log("Can't find a collision-free placement!");
        }
        
        

        return position;

    }

    public static Vector3 GridPosition(
        float xMin,
        float xMax,
        float zMin,
        float zMax,
        float yVal,
        int index,
        int numAgents
        )
    { // Untested copilot code
        if (numAgents == 1)
        {
            return new Vector3(
                (xMin + xMax) / 2, 
                yVal,
                (zMin + zMax) / 2
            );
        }
        int gridSide = Mathf.CeilToInt(Mathf.Sqrt(numAgents));
        var x = xMin + (index % gridSide) * (xMax - xMin) / (gridSide - 1);
        var z = zMin + (index / gridSide) * (zMax - zMin) / (gridSide - 1);
        var pos = new Vector3(x, yVal, z);
        // Debug.Log($"Grid position: {pos} with gridside {gridSide}, index {index}, numAgents {numAgents}, single diff {(xMax - xMin)}");
        return pos;
    }
    
    /// <summary>
    /// Converts the given decimal number to the numeral system with the
    /// specified radix (in the range [2, 36]).
    /// </summary>
    /// <param name="decimalNumber">The number to convert.</param>
    /// <param name="radix">The radix of the destination numeral system (in the range [2, 36]).</param>
    /// <returns></returns>
    public static string DecimalToArbitrarySystem(long decimalNumber, int radix)
    {
        const int BitsInLong = 64;
        const string Digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

        if (radix < 2 || radix > Digits.Length)
            throw new ArgumentException("The radix must be >= 2 and <= " + Digits.Length.ToString());

        if (decimalNumber == 0)
            return "0";

        int index = BitsInLong - 1;
        long currentNumber = Math.Abs(decimalNumber);
        char[] charArray = new char[BitsInLong];

        while (currentNumber != 0)
        {
            int remainder = (int)(currentNumber % radix);
            charArray[index--] = Digits[remainder];
            currentNumber = currentNumber / radix;
        }

        string result = new String(charArray, index + 1, BitsInLong - index - 1);
        if (decimalNumber < 0)
        {
            result = "-" + result;
        }

        return result;
    }

    // public static Vector3 gridPlacement

    // public static float[] GetColliderInfo(Transform baseTransform, Collider collider, bool relative = true)
    // {
    //     
    //     var rigidbody = collider.GetComponent<Rigidbody>();
    //     var transform = collider.transform;
    //     
    //     var pos = transform.localPosition;
    //     var velocity = rigidbody.velocity;
    //     
    //     if (relative)
    //     {
    //         var rotation = baseTransform.localRotation;
    //         pos = Quaternion.Inverse(rotation) * (pos - baseTransform.localPosition);
    //         velocity = Quaternion.Inverse(rotation) * velocity;
    //     }
    //     
    //
    //     return new[] {pos.x, pos.z, velocity.x, velocity.z};
    // }

    // public static float[] GetPredatorPreyInfo(Transform baseTransform, Collider collider)
    // {
    //     var rigidbody = collider.GetComponent<Rigidbody>();
    //     var transform = collider.transform;
    //     var rotation = baseTransform.localRotation;
    //     var pos = Quaternion.Inverse(rotation) * (transform.localPosition - baseTransform.localPosition);
    //     var velocity = Quaternion.Inverse(rotation) * rigidbody.velocity;
    //     var type = Convert.ToSingle(collider.name.Contains("Predator"));
    //
    //     return new[] {pos.x, pos.z, velocity.x, velocity.z, type};
    // }

    [Pure]
    public static float Flood(Vector3 x, Vector3 xMin, Vector3 xMax)
    {
        return Vector3.Min(x - xMin, Vector3.zero).magnitude
            + Vector3.Max(x - xMax, Vector3.zero).magnitude;
    }

    public static Func<Vector2, Vector2> Apply2(Func<float, float> func)
    {
        return v => new Vector2(func(v.x), func(v.y));
    }

    public static Func<Vector3, Vector3> Apply3(Func<float, float> func)
    {
        return v => new Vector3(func(v.x), func(v.y));
    }
    
    public static Vector2 SquashUniform(Vector2 vector)
    {
        // var angle = Vector2.Angle(Vector2.right, vector);

        // tanh(r) * R * e_1
        var result = vector.normalized * MathF.Tanh(vector.magnitude);
        // var result = MathF.Tanh(vector.magnitude) * (Quaternion.AngleAxis(angle, Vector3.forward) * Vector2.right);
        
        // var rotation = Quaternion.Euler(angle, 0, 0);
        // // (R^T o Tanh. o R) (v)
        // var result = Quaternion.Inverse(rotation) * Apply3(MathF.Tanh) (rotation * vector);
        
        return result;
    }

    public static string MakeMessage(Dictionary<string, float> stats)
    {
        var statStrings = stats.Select(kv => $"{kv.Key} {kv.Value}");
        var msg = string.Join('\n', statStrings);
        return msg;
    }

    public static Vector3 GetNoise(float scale)
    {
        var x = Random.Range(-1f, 1f);
        var z = Random.Range(-1f, 1f);
        return new Vector3(x, 0, z) * scale;
    }

    public static bool Visible(Transform baseTransform, Transform targetTransform, float minCosine)
    {
        var direction = targetTransform.localPosition - baseTransform.localPosition;
        var cosine = Vector3.Dot(baseTransform.forward, direction.normalized);
        return cosine >= minCosine;
    }

    public static (float, float) EnergyUsage(Vector3 velocity, Vector3 previousVelocity, float e_s, float e_w, float dt)
    {
        var v = velocity.magnitude;
        var v0 = previousVelocity.magnitude;
        
        var simpleEnergy = e_s * dt + e_w * v * v * dt;

        var acceleration = (velocity - previousVelocity) / dt;
        
        // var complexEnergy = e_s * dt + v*v - (1 - e_w * dt) * Vector3.Dot(previousVelocity, velocity);

        var complexEnergy = e_s * dt +
                            Mathf.Abs(Vector3.Dot(velocity, acceleration)
                                      + e_w * Vector3.Dot(previousVelocity, velocity))
                            * dt;
        
        // TODO: previous and current velocities are computed incorrectly, current velocity doesn't consider what happens due to physics
        // Maybe compute this based on the positions?
        // Debug.Log($"Velocity: {v}, Previous Velocity: {v0}, Simple Energy: {simpleEnergy}, Complex Energy: {complexEnergy}");

        return (simpleEnergy, complexEnergy);
    }
    
    public static float FlatDistance(Vector3 a, Vector3 b)
    {
        var dx = a.x - b.x;
        var dz = a.z - b.z;
        return MathF.Sqrt(dx * dx + dz * dz);
    }
    
    public static float EnergyHeuristic(Vector3 position, Vector3 target, float e_s, float e_w)
    {
        var distance = FlatDistance(position, target);
        var speed = Mathf.Sqrt(e_s / e_w);
        var time = distance / speed;
        
        return e_s * time + e_w * speed * speed * time;
    }
}
