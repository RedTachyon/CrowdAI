using System;
using System.Collections.Generic;
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

    public static float[] GetColliderInfo(Transform baseTransform, Collider collider)
    {
        var rigidbody = collider.GetComponent<Rigidbody>();
        var transform = collider.transform;
        var rotation = baseTransform.localRotation;
        var pos = Quaternion.Inverse(rotation) * (transform.localPosition - baseTransform.localPosition);
        var velocity = Quaternion.Inverse(rotation) * rigidbody.velocity;

        return new[] {pos.x, pos.z, velocity.x, velocity.z};
    }
    
}
