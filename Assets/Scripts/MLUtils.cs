using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Random = UnityEngine.Random;

public class MLUtils
{

    public static Vector3 NoncollidingPosition(
        float min,
        float max,
        float yVal,
        Transform excludes,
        int maxTries = 10,
        float threshold = 0.5f)
    {
        Vector3 position = new Vector3(
            Random.Range(min, max), 
            yVal,
            Random.Range(min, max)
        );
        
        for (var i = 0; i < maxTries; i++)
        {

            var valid = true;
            foreach (Transform t in excludes)
            {
                if ((t.localPosition - position).magnitude < threshold)
                {
                    valid = false;
                    break;
                }
            }

            if (valid)
            {
                break;
            }
            
            position = new Vector3(
                Random.Range(min, max), 
                yVal,
                Random.Range(min, max)
            );
            

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
    
    
}
