using System.Collections.Generic;
using System.Linq;
using UnityEngine;

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
    
    
}
