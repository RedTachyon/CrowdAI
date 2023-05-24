using System;
using System.Collections.Generic;
using Dynamics;
using Initializers;
using Managers;
using Observers;
using Rewards;
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
    
    [Header("Agent settings")]
    
    [Range(1, 100)]
    public int numAgents = 10;
    public static int NumAgents => Mathf.RoundToInt(Get("num_agents", Instance.numAgents));

    public DynamicsEnum dynamics = DynamicsEnum.CartesianVelocity;
    public static DynamicsEnum Dynamics => Enum.Parse<DynamicsEnum>(Get("dynamics", Instance.dynamics.ToString()));

    public ObserversEnum observer = ObserversEnum.Absolute;
    public static ObserversEnum Observer => Enum.Parse<ObserversEnum>(Get("observer", Instance.observer.ToString()));
    
    public InitializerEnum initializer = InitializerEnum.Random;
    public static InitializerEnum Initializer => Enum.Parse<InitializerEnum>(Get("initializer", Instance.initializer.ToString()));
    
    public RewardersEnum rewarder = RewardersEnum.BaseRewarder;
    public static RewardersEnum Rewarder => Enum.Parse<RewardersEnum>(Get("rewarder", Instance.rewarder.ToString()));
    
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
    
    public bool randomMass = false;
    public static bool RandomMass => Convert.ToBoolean(Get("random_mass", Instance.randomMass ? 1f : 0f));
    
    public bool randomEnergy = false;
    public static bool RandomEnergy => Convert.ToBoolean(Get("random_energy", Instance.randomEnergy ? 1f : 0f));
    
    
    public bool sharedGoal;
    public static bool SharedGoal => Convert.ToBoolean(Get("shared_goal", Instance.sharedGoal ? 1f : 0f));
    
    [Header("Physics")]
    public float maxSpeed = 2f;
    public static float MaxSpeed => Get("max_speed", Instance.maxSpeed);
    
    public float maxAcceleration = 5f;
    public static float MaxAcceleration => Get("max_acceleration", Instance.maxAcceleration);

    public float rotationSpeed = 3f;
    public static float RotationSpeed => Get("rotation_speed", Instance.rotationSpeed);
    

    [Header("Unified reward settings")] 
    
    public float rewardBMR = 1f;
    public static float RewBMR => Get("r_bmr", Instance.rewardBMR);
    
    public float rewardDrag = 1f;
    public static float RewDrag => Get("r_drag", Instance.rewardDrag);
    
    public float rewardDynamics = 1f;
    public static float RewDyn => Get("r_dynamics", Instance.rewardDynamics);
    
    public float rewardPotential = 1f;
    public static float RewPot => Get("r_potential", Instance.rewardPotential);
    
    public float rewardDiffPotential = 1f;
    public static float RewDiffPot => Get("r_diff_potential", Instance.rewardDiffPotential);

    public float rewardSpeedMatching = 1f;
    public static float RewSpeed => Get("r_s_matching", Instance.rewardSpeedMatching);
    
    public float rewardSpeedMatchingExp = 1f;
    public static float RewSpeedExp => Get("r_s_matching_exp", Instance.rewardSpeedMatchingExp);

    public float rewardSpeeding = 1f;
    public static float RewSpeeding => Get("r_speeding", Instance.rewardSpeeding);
    
    public float rewardExpVelocityMatching = 1f;
    public static float RewExpVel => Get("r_exp_v_matching", Instance.rewardExpVelocityMatching);
    
    public float rewardExpVelocitySigma = 1f;
    public static float RewExpVelSigma => Get("r_exp_v_sigma", Instance.rewardExpVelocitySigma);
    
    public float rewardVelocityMatching = 1f;
    public static float RewVel => Get("r_v_matching", Instance.rewardVelocityMatching);
    
    public float rewardGoal = 1f;
    public static float RewGoal => Get("r_goal", Instance.rewardGoal);
    
    public float rewardFinal = 1f;
    public static float RewFinal => Get("r_final", Instance.rewardFinal);
    
    


    // Everything else
    [Header("Observation settings")]
    public float sightRadius = 10f;
    public static float SightRadius => Get("sight_radius", Instance.sightRadius);
    
    public int sightAgents = 10;
    public static int SightAgents => Mathf.RoundToInt(Get("sight_agents", Instance.sightAgents));
    
    public float sightAngle = 180f;
    public static float SightAngle => Get("sight_angle", Instance.sightAngle);
    public static float MinCosine => Mathf.Cos(SightAngle * Mathf.Deg2Rad);
    
    public int raysPerDirection = 10; // Only at launch
    public static int RaysPerDirection => Mathf.RoundToInt(Get("rays_per_direction", Instance.raysPerDirection));
    
    public float rayLength = 10f;
    public static float RayLength => Get("ray_length", Instance.rayLength);
    
    public float rayDegrees = 90f;
    public static float RayDegrees => Get("ray_degrees", Instance.rayDegrees);
    
    // Whether rays should hit agents
    public bool rayAgentVision = false;
    public static bool RayAgentVision => Convert.ToBoolean(Get("ray_agent_vision", Instance.rayAgentVision ? 1f : 0f));
    
    public bool destroyRaycasts = false; // Only at launch
    public static bool DestroyRaycasts => Convert.ToBoolean(Get("destroy_raycasts", Instance.destroyRaycasts ? 1f : 0f));
    


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

    public bool showAttention = false;
    public static bool ShowAttention => Convert.ToBoolean(Get("show_attention", Instance.showAttention ? 1f : 0f));
    
    public bool backwardsAllowed = true;
    public static bool BackwardsAllowed => Convert.ToBoolean(Get("backwards_allowed", Instance.backwardsAllowed ? 1f : 0f));
    
    
     [Header("Reward settings")]
    public float potential = 1f;
    public static float Potential => Get("potential", Instance.potential);

    public float goal = 10f;
    public static float Goal => Get("goal", Instance.goal);
    
    public float collision = -0.05f;
    public static float Collision => Get("collision", Instance.collision);
    
    public float stepReward = -0.005f;
    public static float StepReward => Get("step_reward", Instance.stepReward);
    

    public float comfortSpeed = 1.33f;
    public static float ComfortSpeed => Get("comfort_speed", Instance.comfortSpeed);
    
    public float comfortSpeedWeight = -0.75f;
    public static float ComfortSpeedWeight => Get("comfort_speed_weight", Instance.comfortSpeedWeight);

    public float comfortSpeedExponent = 1.0f;
    public static float ComfortSpeedExponent => Get("comfort_speed_exponent", Instance.comfortSpeedExponent);

    // Unused
    
    public float standstillWeight = 0f;
    public static float StandstillWeight => Get("standstill_weight", Instance.standstillWeight);

    public float standstillExponent = 0f;
    public static float StandstillExponent => Get("standstill_exponent", Instance.standstillExponent);
    
    public float goalSpeedThreshold = 0f;
    public static float GoalSpeedThreshold => Get("goal_speed_threshold", Instance.goalSpeedThreshold);

    
    public float comfortDistance = 0f;
    public static float ComfortDistance => Get("comfort_distance", Instance.comfortDistance);
    
    public float comfortDistanceWeight = 0f;
    public static float ComfortDistanceWeight => Get("comfort_distance_weight", Instance.comfortDistanceWeight);


    public float familyGoalRadius = 0.5f;
    public static float FamilyGoalRadius => Get("family_goal_radius", Instance.familyGoalRadius);

    // Energy rewarder
    
    public float energyWeight = 1f;
    public static float EnergyWeight => Get("energy_weight", Instance.energyWeight);
    
    public float finalEnergyWeight = 1f;
    public static float FinalEnergyWeight => Get("final_energy_weight", Instance.finalEnergyWeight);

    public float potentialEnergyScale = 2f;
    public static float PotentialEnergyScale => Get("potential_energy_scale", Instance.potentialEnergyScale);
    
    public bool useComplexEnergy = true;
    public static bool UseComplexEnergy => Convert.ToBoolean(Get("complex_energy", Instance.useComplexEnergy ? 1f : 0f));
    
    // Alignment
    
    public float alignmentWeight = 1f;
    public static float AlignmentWeight => Get("alignment_weight", Instance.alignmentWeight);
        


    private static float Get(string name, float defaultValue)
    {
        return Academy.Instance.EnvironmentParameters.GetWithDefault(name, defaultValue);
    }

    private static string Get(string name, string defaultValue)
    {
        return Manager.Instance.StringChannel.GetWithDefault(name, defaultValue);
    }
}