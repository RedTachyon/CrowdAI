using System;
using System.Collections.Generic;
using Dynamics;
using Managers;
using Observers;
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
      // // /// //  
     // REWARD //
    // // // //
    public float potential = 0.4f;
    public static float Potential => Get("potential", Instance.potential);

    public float goal = 1.0f;
    public static float Goal => Get("goal", Instance.goal);
    
    public float collision = -0.3f;
    public static float Collision => Get("collision", Instance.collision);
    
    public float goalSpeedThreshold = 1e-1f;
    public static float GoalSpeedThreshold => Get("goal_speed_threshold", Instance.goalSpeedThreshold);

    public float comfortSpeed = 1.0f;
    public static float ComfortSpeed => Get("comfort_speed", Instance.comfortSpeed);
    
    public float comfortSpeedWeight = 0.1f;
    public static float ComfortSpeedWeight => Get("comfort_speed_weight", Instance.comfortSpeedWeight);

    public float comfortSpeedExponent = 1.0f;
    public static float ComfortSpeedExponent => Get("comfort_speed_exponent", Instance.comfortSpeedExponent);

    public float comfortDistance = 1.5f;
    public static float ComfortDistance => Get("comfort_distance", Instance.comfortDistance);
    
    public float comfortDistanceWeight = 1.0f;
    public static float ComfortDistanceWeight => Get("comfort_distance_weight", Instance.comfortDistanceWeight);

    public float stepReward = -0.05f;
    public static float StepReward => Get("step_reward", Instance.stepReward);
    
    // Energy parameters
    public float e_s = 2.23f;
    public static float E_s => Get("e_s", Instance.e_s);

    public float e_w = 1.26f;
    public static float E_w => Get("e_w", Instance.e_w);
    
    // Everything else
    
    public float sightRadius = 5f;
    public static float SightRadius => Get("radius", Instance.sightRadius);
    
    public int sightAgents = 10;
    public static int SightAgents => Mathf.RoundToInt(Get("sight_agents", Instance.sightAgents));
    
    // Spawn
    
    public float spawnNoiseScale = 0.5f;
    public static float SpawnNoiseScale => Get("spawn_noise_scale", Instance.spawnNoiseScale);
    
    // Meta

    public bool evaluationMode = false;
    public static bool EvaluationMode => Convert.ToBoolean(Get("evaluation_mode", Instance.evaluationMode ? 1f : 0f));

    public string savePath = "";
    public static string SavePath => Get("save_path", Instance.savePath);

    public DynamicsEnum dynamics = DynamicsEnum.CartesianVelocity;
    public static DynamicsEnum Dynamics => Enum.Parse<DynamicsEnum>(Get("dynamics", Instance.dynamics.ToString()));

    public ObserversEnum observer = ObserversEnum.Absolute;
    public static ObserversEnum Observer => Enum.Parse<ObserversEnum>(Get("observer", Instance.observer.ToString()));

    private static float Get(string name, float defaultValue)
    {
        return Academy.Instance.EnvironmentParameters.GetWithDefault(name, defaultValue);
    }

    private static string Get(string name, string defaultValue)
    {
        return Manager.Instance.StringChannel.GetWithDefault(name, defaultValue);
    }
}