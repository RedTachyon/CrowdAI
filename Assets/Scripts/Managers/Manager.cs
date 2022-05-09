using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Agents;
using Initializers;
using Newtonsoft.Json;
using Unity.Barracuda;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Managers
{

    public class Manager : MonoBehaviour
    {
        [Range(1, 100)]
        public int numAgents = 1;
        public string dataFileName;

        [Range(1, 1000)]
        public int maxStep = 500;

        [Range(1, 10)] public int decisionFrequency = 1;
        
        protected Dictionary<Transform, bool> _finished;
        internal int Timestep;
        public StatsCommunicator statsCommunicator;

        public StringChannel StringChannel;
        
        protected SimpleMultiAgentGroup _agentGroup;

        protected bool _initialized;
        protected int _episodeNum;

        protected float[,,] _positionMemory;
        protected float[] _timeMemory;

        public Transform AllObstacles;

        [NonSerialized]
        public Vector3 goalScale;

        protected static Manager _instance;
        public static Manager Instance => _instance;

        public void Awake()
        {
            if (_instance != null && _instance != this)
            {
                Destroy(gameObject);
            }
            else
            {
                _instance = this;
            }
            
            _finished = new Dictionary<Transform, bool>();
            Academy.Instance.OnEnvironmentReset += ResetEpisode;

            var agent = GetComponentInChildren<AgentBasic>();
            goalScale = agent.goal.localScale;
            _agentGroup = new SimpleMultiAgentGroup();

            foreach (Transform _agent in transform)
            {
                _agentGroup.RegisterAgent(agent.GetComponent<Agent>());
            }
            
            StringChannel = new StringChannel();
            SideChannelManager.RegisterSideChannel(StringChannel);

            _episodeNum = 0;
            

        }

        private void Start()
        {
            if (false)
            {
                // This is disabled for deployment, but it's useful to run sometimes during development
                var jsonParams = JsonUtility.ToJson(Params.Instance);
                Debug.Log("Writing default params to file");
                Debug.Log(jsonParams);
                File.WriteAllText("params.json", jsonParams);
            }
        }

        public virtual void ResetEpisode()
        {

            Debug.Log("ResetEpisode");

            _episodeNum++;
            _initialized = true;
            
            numAgents = Params.NumAgents;

            _positionMemory = new float[numAgents, maxStep * decisionFrequency, 2];
            _timeMemory = new float[maxStep * decisionFrequency];

            var currentNumAgents = transform.childCount;
            var agentsToAdd = numAgents - currentNumAgents;
            
            Debug.Log($"Number of children: {currentNumAgents}");

            // Activate the right amount of agents
            for (var i = 0; i < currentNumAgents; i++)
            {
                var active = i < numAgents;
                var currentAgent = transform.GetChild(i);
                currentAgent.gameObject.SetActive(active);
                var currentGoal = currentAgent.GetComponent<AgentBasic>().goal;
                currentGoal.gameObject.SetActive(active);

                Agent agent = currentAgent.GetComponent<Agent>();

                if (active)
                {
                    _agentGroup.RegisterAgent(agent);
                }
                else
                {
                    _agentGroup.UnregisterAgent(agent);
                }
            
            }
        
            var baseAgent = GetComponentInChildren<AgentBasic>();
            var baseGoal = baseAgent.goal;

            // If necessary, add some more agents
            if (agentsToAdd > 0) Debug.Log($"Creating {agentsToAdd} new agents");
        
            for (var i = currentNumAgents; i < numAgents; i++)
            {
                var newAgent = Instantiate(baseAgent, transform);
                var newGoal = Instantiate(baseGoal, baseGoal.parent);
            
                newAgent.GetComponent<AgentBasic>().goal = newGoal;
                newAgent.name = baseAgent.name + $" ({i})";
                newGoal.name = baseGoal.name + $" ({i})";
            }
        
            // Give'em some color
            int agentIdx = 0;
            foreach (Transform agentTransform in transform)
            {
                var agent = agentTransform.GetComponent<AgentBasic>();
                agent.SetColor(ColorMap.GetColor(agentIdx), true);
                
                // Choose a random mass
                // var mass = Random.Range(0.5f, 1.5f);
                var mass = 1f;
                agent.mass = mass;
                agentTransform.localScale *= Mathf.Pow(mass, 0.333333f);

                agentTransform.position = new Vector3(0f, agentTransform.localScale.y, 0f);

                agentIdx++;
            }
            
            // Remove all obstacles
            foreach (Transform obstacle in AllObstacles)
            {
                obstacle.gameObject.SetActive(false);
            }
            
            // Find the right locations for all agents
            Debug.Log($"Total agents: {transform.childCount}");
            IInitializer initializer = Mapper.GetInitializer(Params.Initializer, dataFileName);
            initializer.PlaceAgents(transform, Params.SpawnScale, initializer.GetObstacles());


            
            // Initialize stats
            _finished.Clear();

            Timestep = 0;

            foreach (Transform agent in transform)
            {
                _finished[agent] = false;
                agent.GetComponent<AgentBasic>().OnEpisodeBegin();
            }

        }
        public virtual void ReachGoal(Agent agent)
        {
            _finished[agent.GetComponent<Transform>()] = true;
            // agent.GetComponent<AgentBasic>().CollectedGoal = true;

        }

        private void WriteTrajectory()
        {
            Debug.Log("Trying to save a trajectory");
            string fullSavePath;
            if (Params.SavePath != "DEFAULT")
            {
                fullSavePath = Params.SavePath;
            }
            else
            {
                if (!Directory.Exists("output"))
                {
                    Directory.CreateDirectory("output");
                }

                fullSavePath = Path.Combine("output", $"trajectory_{_episodeNum}.json");
                // var savePath = $"output/trajectory_{_episodeNum}.json";
            }
            
            Debug.Log($"Writing to {fullSavePath}");
            var data = new TrajectoryData(_timeMemory, _positionMemory);
            var json = JsonConvert.SerializeObject(data);
            File.WriteAllText(fullSavePath, json);
        }

        private void FixedUpdate()
        {
            if (!_initialized) return;
            
            // Debug.Log(Time.fixedDeltaTime);
            Dictionary<string, float> episodeStats = null;
            // Reset the episode if time runs out, or if all agents have reached their goals (and early finish is enabled)
            if (Timestep >= maxStep * decisionFrequency || (Params.EarlyFinish && _finished.Values.All(x => x)))
            {
                episodeStats = GetEpisodeStats();
                if (Params.SavePath != "") WriteTrajectory();
                else Debug.Log("Oops, not saving anything");

                Debug.Log("Resetting");
                
                _agentGroup.EndGroupEpisode();
                ResetEpisode();
            }

            CollectStats(episodeStats);

            ///////////////////////
            // Log the positions //
            ///////////////////////
            
            var agentIdx = 0;
            // var decisionTime = Time / decisionFrequency;
            foreach (Transform agent in transform)
            {
                if (!agent.gameObject.activeInHierarchy) continue;
                
                var localPosition = agent.localPosition;
                _positionMemory[agentIdx, Timestep, 0] = localPosition.x;
                _positionMemory[agentIdx, Timestep, 1] = localPosition.z;

                agentIdx++;
            }

            _timeMemory[Timestep] = Timestep * Time.fixedDeltaTime;
            

            /////////////////////
            // Request actions //
            /////////////////////

            if (Timestep % decisionFrequency == 0)
            {
                // Debug.Log($"Timestep: {Timestep}; Time: {Timestep * Time.fixedDeltaTime}");

                foreach (Transform agent in transform)
                {
                    agent.GetComponent<Agent>().RequestDecision();
                }
            } else
            {
                foreach (Transform agent in transform)
                {
                    agent.GetComponent<Agent>().RequestAction();
                }
            }

            Timestep++;
        
            // Debug.Log(Time);

            

        }

        private Dictionary<string, float> GetEpisodeStats()
        {
            
            var energies = new List<float>();
            var distances = new List<float>();
            var successes = new List<float>();
            var numAgents = 0;
            
            foreach (Transform agent in transform)
            {
                energies.Add(agent.GetComponent<AgentBasic>().energySpent);
                distances.Add(agent.GetComponent<AgentBasic>().distanceTraversed);
                successes.Add(agent.GetComponent<AgentBasic>().CollectedGoal ? 1f : 0f);
                numAgents++;
            }

            var stats = new Dictionary<string, float>
            {
                ["e_energy"] = energies.Average(),
                ["e_distance"] = distances.Average(),
                ["e_success"] = successes.Average(),
            };
            
            foreach (Transform agentTransform in transform)
            {
                var agent = agentTransform.GetComponent<AgentBasic>();
                foreach (var rewardPart in agent.rewardParts)
                {
                    var keyname = $"e_reward_{rewardPart.Key}";
                    if (stats.ContainsKey(keyname))
                    {
                        stats[keyname] += rewardPart.Value / numAgents;
                    } else
                    {
                        stats[keyname] = rewardPart.Value / numAgents;
                    }
                }
            }

            return stats;
        } 


        private void CollectStats(Dictionary<string, float> episodeStats = null)
        {
            var distances = new List<float>();
            var speeds = new List<float>();
            var dones = new List<float>();
            var collisions = new List<int>();
        
            foreach (Transform agent in transform)
            {
                if (!agent.gameObject.activeSelf) continue;
                // Get distance from goal
                var agentPosition = agent.localPosition;
                var goalPosition = agent.GetComponent<AgentBasic>().goal.localPosition;

                var distance = Vector3.Distance(agentPosition, goalPosition);
                distances.Add(distance);
            
                // Get speed
                var speed = agent.GetComponent<Rigidbody>().velocity.magnitude;
                speeds.Add(speed);
            
                // Debug.Log($"Stats from agent {agent.name}");
                // Fraction of agents  that finished already
                dones.Add(_finished[agent] ? 1f : 0f);
                // Debug.Log(_finished[agent]);
            
                collisions.Add(agent.GetComponent<AgentBasic>().Collision);

            }
            var meanDist = distances.Average();
            var meanSpeed = speeds.Average();
            var finished =  dones.Average();
            var collision = (float) collisions.Average();
            // TODO: at some point uniformize e_name and m_name
            episodeStats ??= new Dictionary<string, float>();
            episodeStats["mean_distance"] = meanDist;
            episodeStats["mean_speed"] = meanSpeed;
            episodeStats["mean_finished"] = finished;
            episodeStats["mean_collision"] = collision;
            
        
            // Debug.Log(collision);

            var message = MLUtils.MakeMessage(episodeStats);
            
            // var message = $"mean_dist {meanDist}\nmean_speed {meanSpeed}\nmean_finish {finished}\nmean_collision {collision}";
            statsCommunicator.StatsChannel.SendMessage(message);
            // Debug.Log("Message allegedly sent");
        }

        // public InitializerEnum GetMode()
        // {
        //     var val = Academy.Instance.EnvironmentParameters.GetWithDefault("mode", -1f);
        //     InitializerEnum currentMode;
        //     if (val < -0.5f) // == -1f 
        //     {
        //         currentMode = mode;
        //     }
        //     else if (val < 0.5f) // == 0f
        //     {
        //         currentMode = InitializerEnum.Random;
        //     } 
        //     else if (val < 1.5f) // == 1f
        //     {
        //         currentMode = InitializerEnum.Circle;
        //     }
        //     else if (val < 2.5f) // == 2f
        //     {
        //         currentMode = InitializerEnum.Crossway;
        //     }
        //     else
        //     {
        //         currentMode = InitializerEnum.JsonInitializer;
        //     }
        //
        //     return currentMode;
        // }

        // public int GetNumAgents()
        // {
        //     var val = Academy.Instance.EnvironmentParameters.GetWithDefault("agents", -1f);
        //     Debug.Log("Requested number of agents: " + val);
        //     int agents;
        //     agents = val < 0 ? numAgents : Mathf.RoundToInt(val);
        //
        //     var newNumAgents = Math.Max(agents, 1);
        //     Debug.Log("New number of agents: " + newNumAgents);
        //     return newNumAgents;
        // }

        private void OnDestroy()
        {
            if (Academy.IsInitialized)
            {
                SideChannelManager.UnregisterSideChannel(StringChannel);
            }
        }
    }
}