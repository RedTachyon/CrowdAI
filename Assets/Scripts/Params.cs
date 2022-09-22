using System;
using System.Collections.Generic;
using Dynamics;
using Initializers;
using Managers;
using Observers;
using Unity.MLAgents;
using UnityEngine;

[Serializable]
public class Params : MonoBehaviour
{

    private static Params _instance;
    public static Params Instance => _instance;
    
    
    private void Awake()
    {
        Debug.Log("Awaking params");
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
    
    [Header("Reward settings")]
    public float potential = 0.4f;
    public static float Potential => Get("potential", Instance.potential);

    public float goal = 1.0f;
    public static float Goal => Get("goal", Instance.goal);
    
    public float collision = -0.3f;
    public static float Collision => Get("collision", Instance.collision);
    
    public float stepReward = -0.005f;
    public static float StepReward => Get("step_reward", Instance.stepReward);
    
    public float standstillWeight = -0.1f;
    public static float StandstillWeight => Get("standstill_weight", Instance.standstillWeight);

    public float standstillExponent = 1f;
    public static float StandstillExponent => Get("standstill_exponent", Instance.standstillExponent);
    
    public float goalSpeedThreshold = 1e-1f;
    public static float GoalSpeedThreshold => Get("goal_speed_threshold", Instance.goalSpeedThreshold);

    public float comfortSpeed = 1.0f;
    public static float ComfortSpeed => Get("comfort_speed", Instance.comfortSpeed);
    
    public float comfortSpeedWeight = -0.75f;
    public static float ComfortSpeedWeight => Get("comfort_speed_weight", Instance.comfortSpeedWeight);

    public float comfortSpeedExponent = 1.0f;
    public static float ComfortSpeedExponent => Get("comfort_speed_exponent", Instance.comfortSpeedExponent);

    public float comfortDistance = 1.5f;
    public static float ComfortDistance => Get("comfort_distance", Instance.comfortDistance);
    
    public float comfortDistanceWeight = 1.0f;
    public static float ComfortDistanceWeight => Get("comfort_distance_weight", Instance.comfortDistanceWeight);
    
    // Energy parameters
    [Header("Energy parameters")]
    public float e_s = 2.23f;
    public static float E_s => Get("e_s", Instance.e_s);

    public float e_w = 1.26f;
    public static float E_w => Get("e_w", Instance.e_w);

    public bool energyComplex = true;
    public static bool EnergyComplex => Convert.ToBoolean(Get("energy_complex", Instance.energyComplex ? 1f : 0f)); 
    
    // Everything else
    [Header("Observation settings")]
    public float sightRadius = 10f;
    public static float SightRadius => Get("sight_radius", Instance.sightRadius);
    
    public int sightAgents = 10;
    public static int SightAgents => Mathf.RoundToInt(Get("sight_agents", Instance.sightAgents));
    
    public float sightAngle = 180f;
    public static float SightAngle => Get("sight_angle", Instance.sightAngle);
    public static float MinCosine => Mathf.Cos(SightAngle * Mathf.Deg2Rad);
    
    public bool sightAcceleration = false;
    public static bool SightAcceleration => Convert.ToBoolean(Get("sight_acceleration", Instance.sightAcceleration ? 1f : 0f));

    public int raysPerDirection = 10; // Only at launch
    public static int RaysPerDirection => Mathf.RoundToInt(Get("rays_per_direction", Instance.raysPerDirection));
    
    public float rayLength = 40f;
    public static float RayLength => Get("ray_length", Instance.rayLength);
    
    public float rayDegrees = 90f;
    public static float RayDegrees => Get("ray_degrees", Instance.rayDegrees);
    
    // Whether rays should hit agents
    public bool rayAgentVision = false;
    public static bool RayAgentVision => Convert.ToBoolean(Get("ray_agent_vision", Instance.rayAgentVision ? 1f : 0f));
    
    public bool destroyRaycasts = false; // Only at launch
    public static bool DestroyRaycasts => Convert.ToBoolean(Get("destroy_raycasts", Instance.destroyRaycasts ? 1f : 0f));
    
    // Spawn
    [Header("Spawn settings")]
    public float spawnNoiseScale = 0.5f;
    public static float SpawnNoiseScale => Get("spawn_noise_scale", Instance.spawnNoiseScale);

    public float spawnScale = 4f;
    public static float SpawnScale => Get("spawn_scale", Instance.spawnScale);

    public bool gridSpawn = true;
    public static bool GridSpawn => Convert.ToBoolean(Get("grid_spawn", Instance.gridSpawn ? 1f : 0f));

    public float groupSpawnScale = 1.5f;
    public static float GroupSpawnScale => Get("group_spawn_scale", Instance.groupSpawnScale);
    
    public bool enableObstacles = true;
    public static bool EnableObstacles => Convert.ToBoolean(Get("enable_obstacles", Instance.enableObstacles ? 1f : 0f));
    
    public float blockScale = 1f;
    public static float BlockScale => Get("block_scale", Instance.blockScale);
    
    
    public bool sharedGoal;
    public static bool SharedGoal => Convert.ToBoolean(Get("shared_goal", Instance.sharedGoal ? 1f : 0f));

    // Meta
    [Header("Meta settings")]
    public bool evaluationMode = false;
    public static bool EvaluationMode => Convert.ToBoolean(Get("evaluation_mode", Instance.evaluationMode ? 1f : 0f));

    public string savePath = "";
    public static string SavePath => Get("save_path", Instance.savePath);
    
    public bool earlyFinish = false;
    public static bool EarlyFinish => Convert.ToBoolean(Get("early_finish", Instance.earlyFinish ? 1f : 0f));

    public bool niceColors = true;
    public static bool NiceColors => Convert.ToBoolean(Get("nice_colors", Instance.niceColors ? 1f : 0f));
    
    public bool backwardsAllowed = false;
    public static bool BackwardsAllowed => Convert.ToBoolean(Get("backwards_allowed", Instance.backwardsAllowed ? 1f : 0f));
        
    [Header("Agent settings")]
    
    [Range(1, 100)]
    public int numAgents = 10;
    public static int NumAgents => Mathf.RoundToInt(Get("num_agents", Instance.numAgents));

    public DynamicsEnum dynamics = DynamicsEnum.CartesianVelocity;
    public static DynamicsEnum Dynamics => Enum.Parse<DynamicsEnum>(Get("dynamics", Instance.dynamics.ToString()));

    public ObserversEnum observer = ObserversEnum.Absolute;
    public static ObserversEnum Observer => Enum.Parse<ObserversEnum>(Get("observer", Instance.observer.ToString()));
    
    public InitializerEnum initializer = InitializerEnum.Random; // TODO: uniformize mode and initializer
    public static InitializerEnum Initializer => Enum.Parse<InitializerEnum>(Get("mode", Instance.initializer.ToString()));

    private static float Get(string name, float defaultValue)
    {
        return Academy.Instance.EnvironmentParameters.GetWithDefault(name, defaultValue);
    }

    private static string Get(string name, string defaultValue)
    {
        return Manager.Instance.StringChannel.GetWithDefault(name, defaultValue);
    }
}