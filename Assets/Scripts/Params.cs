using System;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;


public class Params : MonoBehaviour
{

    private static Params _instance;
    public static Params Instance => _instance;
    
    
    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
        }
        else
        {
            _instance = this;
        }
    }

    public float potential = 0.4f;
    public static float Potential => Get("potential", Instance.potential);

    public float goal = 1.0f;
    public static float Goal => Get("goal", Instance.goal);
    
    public float collision = -0.3f;
    public static float Collision => Get("collision", Instance.collision);
    
    public float sightRadius = 5f;
    public static float SightRadius => Get("radius", Instance.sightRadius);
    
    public int sightAgents = 10;
    public static int SightAgents => Mathf.RoundToInt(Get("sight_agents", Instance.sightAgents));
    
    public float comfortSpeed = 1.0f;
    public static float ComfortSpeed => Get("comfort_speed", Instance.comfortSpeed);
    
    public float comfortSpeedWeight = 0.1f;
    public static float ComfortSpeedWeight => Get("comfort_speed_weight", Instance.comfortSpeedWeight);
    
    public float comfortDistance = 1.5f;
    public static float ComfortDistance => Get("comfort_distance", Instance.comfortDistance);
    
    public float comfortDistanceWeight = 1.0f;
    public static float ComfortDistanceWeight => Get("comfort_distance_weight", Instance.comfortDistanceWeight);
    
    public bool saveTrajectory = true;
    public static bool SaveTrajectory => Get("save_trajectory", Instance.saveTrajectory ? 1f : 0f) > 0.5f;

    private static float Get(string name, float defaultValue)
    {
        return Academy.Instance.EnvironmentParameters.GetWithDefault(name, defaultValue);
    }
}