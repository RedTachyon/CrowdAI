using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Agents;
using Initializers;
using Newtonsoft.Json;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEditor;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Managers
{

    public class Manager : MonoBehaviour
    {
        private int numAgents = 1;
        public string dataFileName;

        [Range(1, 1000)]
        public int maxStep = 200;

        [Range(1, 10)] public int decisionFrequency = 1;
        
        protected Dictionary<Transform, bool> _finished;
        [Range(1, 2000)] [SerializeField] internal int Timestep;
        [Range(1, 200)] [SerializeField] internal int DecisionTimestep;
        public StatsCommunicator statsCommunicator;

        public StringChannel StringChannel;
        public AttentionChannel AttentionChannel;
        
        protected SimpleMultiAgentGroup _agentGroup;

        protected bool _initialized;
        protected int _episodeNum;

        protected float[,,] _positionMemory;
        protected float[] _timeMemory;
        protected float[,] _goalPosition;
        protected int[] _finishTime;

        public Transform AllObstacles;

        [NonSerialized]
        public Vector3 goalScale;
        
        public int selectedIdx = 0;

        private FamilyAgent _familyAgent;
        
        
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

            goalScale = GetComponentInChildren<AgentBasic>().Goal.localScale;
            // goalScale = agent.goal.localScale;
            _agentGroup = new SimpleMultiAgentGroup();

            foreach (Transform agent in transform)
            {
                _agentGroup.RegisterAgent(agent.GetComponent<Agent>());
            }
            
            StringChannel = new StringChannel();
            SideChannelManager.RegisterSideChannel(StringChannel);
            
            AttentionChannel = new AttentionChannel();
            SideChannelManager.RegisterSideChannel(AttentionChannel);
            

            _episodeNum = 0;
            
            _familyAgent = transform.parent.GetComponentInChildren<FamilyAgent>();
            
            Random.InitState(0);
            

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
            _goalPosition = new float[numAgents, 2];
            _finishTime = new int[numAgents];
            

            var currentNumAgents = transform.childCount;
            var agentsToAdd = numAgents - currentNumAgents;
            
            _agentGroup.Dispose();
            
            Debug.Log($"Number of children: {currentNumAgents}");

            // Go through all existing agents, activate just enough of them
            for (var i = 0; i < currentNumAgents; i++)
            {
                var active = i < numAgents;
                
                var currentAgent = transform.GetChild(i);
                currentAgent.gameObject.SetActive(active);
                var currentGoal = currentAgent.GetComponent<AgentBasic>().Goal;
                currentGoal.gameObject.SetActive(active);

                Agent agent = currentAgent.GetComponent<Agent>();

                if (active)
                {
                    _agentGroup.RegisterAgent(agent);
                }

            }
        
            var baseAgent = GetComponentInChildren<AgentBasic>();
            var baseGoal = baseAgent.Goal;

            // If necessary, add some more agents
            if (agentsToAdd > 0) Debug.Log($"Creating {agentsToAdd} new agents");
        
            for (var i = currentNumAgents; i < numAgents; i++)
            {
                var newAgent = Instantiate(baseAgent, transform);
                var newGoal = Instantiate(baseGoal, baseGoal.parent);
            
                newAgent.GetComponent<AgentBasic>().Goal = newGoal;
                newAgent.name = baseAgent.name + $" ({i})";
                newGoal.name = baseGoal.name + $" ({i})";
            }
        
            // Set up public parameters of all agents
            int agentIdx = 0;
            foreach (var (agentTransform, agent) in ActiveAgentsTransform<AgentBasic>())
            {
                Debug.Log($"Setting up agent {agent.name}");
                agent.SetColor(ColorMap.GetColor(agentIdx), true);
                
                agent.AgentIndex = agentIdx;
                // Choose a random mass
                var mass = Params.RandomMass ? Random.Range(0.5f, 1.5f) : 1f;
                // var mass = 1f;
                
                var e_s = Params.RandomEnergy ? Random.Range(1.5f, 3f) : 2.23f;
                var e_w = Params.RandomEnergy ? Random.Range(1f, 1.5f) : 1.26f;

                var prefSpeed = Mathf.Sqrt(e_s / e_w);
                
                // Choose a random friction
                
                agent.mass = mass;
                agent.e_s = e_s;
                agent.e_w = e_w;
                agent.PreferredSpeed = prefSpeed;
                
                var factor = Mathf.Pow(mass, 0.333333f);
                // var tempScale = agentTransform.localScale;
                // var tempScale = new Vector3(0.4f, 0.875f, 0.4f);
                var tempScale = new Vector3(1f, 1f, 1f);

                // agentTransform.localScale.Scale(new Vector3(factor, 1, factor));
                agentTransform.localScale = new Vector3(factor*tempScale.x, tempScale.y, factor*tempScale.z);

                agentTransform.position = new Vector3(0f, agentTransform.localScale.y, 0f);

                agentIdx++;
            }
            
            // Remove all obstacles
            foreach (Transform obstacle in AllObstacles)
            {
                obstacle.gameObject.SetActive(false);
            }
            
            // Find the right locations for all agents
            Debug.Log($"Total agents: {transform.Cast<Transform>().Count()}");
            IInitializer initializer = Mapper.GetInitializer(Params.Initializer, dataFileName);
            initializer.PlaceAgents(transform, Params.SpawnScale, initializer.GetObstacles());

            var agentIdxGoal = 0;
            // var decisionTime = Time / decisionFrequency;
            foreach (var agent in ActiveAgents<AgentBasic>())
            {
                var localPosition = agent.Goal.localPosition;
                _goalPosition[agentIdxGoal, 0] = localPosition.x;
                _goalPosition[agentIdxGoal, 1] = localPosition.z;
                _finishTime[agentIdxGoal] = -1;

                agentIdxGoal++;
            }
            
            // for (int i = 0; i < numAgents; i++)
            // {
            //     _finishTime[i] = -1;
            // }

            
            // Initialize stats
            _finished.Clear();

            Timestep = 0;
            DecisionTimestep = 0;
            
            _familyAgent?.ResetAgents();

            foreach (var (agentTransform, agent) in ActiveAgentsTransform<AgentBasic>())
            {
                _finished[agentTransform] = false;
                agent.GetComponent<AgentBasic>().OnEpisodeBegin();
                _familyAgent?.AddAgent(agent.GetComponent<AgentBasic>());
            }
            
            _familyAgent?.OnEpisodeBegin();
            // Debug.Log($"Saving a screenshot to {Application.persistentDataPath}");
            // ScreenCapture.CaptureScreenshot("LastScreenshot.png");

        }

        private void FixedUpdate()
        {
            if (!_initialized) return;

            Dictionary<string, float> episodeStats = null;
            var terminal = false;
            // Reset the episode if time runs out, or if all agents have reached their goals (and early finish is enabled)
            // TODO: base termination on decision steps?
            // if (Timestep >= maxStep * decisionFrequency || (Params.EarlyFinish && _finished.Values.All(x => x)))
            if (DecisionTimestep >= maxStep || (Params.EarlyFinish && _finished.Values.All(x => x)))
            {
                foreach (var agent in ActiveAgents<AgentBasic>())
                {
                    agent.AddFinalReward();
                }
                episodeStats = GetEpisodeStats();
                if (Params.SavePath != "") WriteTrajectory();
                else Debug.Log("Oops, not saving anything");

                Debug.Log("Resetting");
                // Debug.Break();
                // return;

                terminal = true;

                _agentGroup.EndGroupEpisode();
                _familyAgent?.EndEpisode();
                ResetEpisode();
            }


            ///////////////////////
            // Log the positions //
            ///////////////////////
            
            var agentIdx = 0;
            // var decisionTime = Time / decisionFrequency;
            foreach (var (agent, _) in ActiveAgentsTransform<AgentBasic>())
            {
                var localPosition = agent.localPosition;
                _positionMemory[agentIdx, Timestep, 0] = localPosition.x;
                _positionMemory[agentIdx, Timestep, 1] = localPosition.z;

                agentIdx++;
            }
            
            // Debug.Log($"Timestep: {Timestep}");

            _timeMemory[Timestep] = Timestep * Time.fixedDeltaTime;
            

            /////////////////////
            // Request actions //
            /////////////////////

            if (Timestep % decisionFrequency == 0)
            {
                // Debug.Log($"Timestep: {Timestep}; Time: {Timestep * Time.fixedDeltaTime}");
                
                // Collect stats only when requesting a decision
                CollectStats(episodeStats);

                _familyAgent?.RequestDecision();
                foreach (var agent in ActiveAgents<Agent>())
                {
                    agent.RequestDecision();
                }
                
                if (Params.ShowAttention)
                {
                    AgentBasic selectedAgent;
                    selectedAgent = transform.GetChild(selectedIdx).GetComponent<AgentBasic>();

#if UNITY_EDITOR
                    if (Application.isEditor && Selection.activeTransform != null && Selection.activeTransform.GetComponent<AgentBasic>() != null)
                    {
                        selectedAgent = Selection.activeTransform.GetComponent<AgentBasic>();
                    }
#endif
                    // var selectedAgent = Selection.activeTransform != null ? Selection.activeTransform.GetComponent<AgentBasic>() : null;
                    // var selectedAgent = transform.GetChild(selectedIdx).GetComponent<AgentBasic>();
                    var neighbors = selectedAgent.neighborsOrder;
                    var attentionValues = AttentionChannel.Attention[selectedIdx];
                    // Debug.Log($"Attention values: {attentionValues}");
                    
                    // Reset all agents' colors
                    foreach (var agent in ActiveAgents<AgentBasic>())
                    {
                        agent.SetColor(new Color(1, 1, 1), true);
                    }
                    
                    // Color the selected agent
                    
                    selectedAgent.SetColor(new Color(0, 1, 0), false);
                    
                    // Color the neighbors
                    
                    foreach ((AgentBasic agent, int attention) in neighbors.Zip(attentionValues, (a, b) => (a.GetComponent<AgentBasic>(), b)))
                    {
                        // agent.SetColor(new Color(0, 0, attention / 100f), false);
                        // Observed agents go from white to blue
                        agent.SetColor(Color.HSVToRGB(220f/255f, 0.2f + 8*attention/100f, 1f), false);
                        // agent.SetColor(Color.HSVToRGB(160f/255f, 1f, 1f), false);
                    }

                }

                DecisionTimestep++;
            } else
            {
                _familyAgent?.RequestAction();
                foreach (var agent in ActiveAgents<Agent>())
                {
                    agent.RequestAction();
                }
            }

        
            // Debug.Log(Time);
            
            // Debug.Log("Before physics");
            Physics.Simulate(Time.fixedDeltaTime);
            // Debug.Log("After physics");
            
            _familyAgent?.TryFinish();

            foreach (var (agent_transform, agent) in ActiveAgentsTransform<AgentBasic>())
            {
                var reward = agent._rewarder.LateReward(agent_transform);
                agent.AddReward(reward);
                
                agent.RecordEnergy();
                
                // if (Timestep % decisionFrequency == 0) Debug.Log($"Reward of agent {agent.name}: {agent.GetCurrentReward()}");
            }

            Timestep++;


        }

        public virtual void ReachGoal(Agent agent)
        {
            var idx = agent.GetComponent<AgentBasic>().AgentIndex;
            _finished[agent.GetComponent<Transform>()] = true;
            if (_finishTime[idx] < 0) _finishTime[idx] = Timestep;
            // agent.GetComponent<AgentBasic>().CollectedGoal = true;

        }

        private Dictionary<string, float> GetEpisodeStats()
        {
            
            var energies = new List<float>();
            var energiesComplex = new List<float>();
            var energiesPlus = new List<float>();
            var energiesComplexPlus = new List<float>();
            var energiesPlusAvg = new List<float>();
            var energiesComplexPlusAvg = new List<float>();
            var distances = new List<float>();
            var successes = new List<float>();
            var numAgents = 0;
            
            foreach (var agent in ActiveAgents<AgentBasic>())
            {
                energies.Add(agent.energySpent);
                energiesComplex.Add(agent.energySpentComplex);
                
                // var finalDistance = MLUtils.FlatDistance(agent.transform.localPosition, agent.Goal.localPosition);

                // var finalEnergy = 2 * Mathf.Sqrt(agent.e_s * agent.e_w * finalDistance);

                var localPosition = agent.transform.localPosition;
                var goalPosition = agent.Goal.localPosition;
                var finalEnergy = MLUtils.EnergyHeuristic(localPosition, goalPosition,
                    agent.e_s, agent.e_w);
                
                var finalEnergyAvg = MLUtils.AverageEnergyHeuristic(localPosition, goalPosition, agent.startPosition,
                    agent.e_s, agent.e_w);
                
                energiesPlus.Add(agent.energySpent + finalEnergy);
                energiesComplexPlus.Add(agent.energySpentComplex + finalEnergy);
                
                energiesPlusAvg.Add(agent.energySpent + finalEnergyAvg);
                energiesComplexPlusAvg.Add(agent.energySpentComplex + finalEnergyAvg);
                
                distances.Add(agent.distanceTraversed);
                successes.Add(agent.CollectedGoal ? 1f : 0f);
                numAgents++;
            }
            // Debug.Log($"NumAgents detected in EpisodeStats: {numAgents}");
            var stats = new Dictionary<string, float>
            {
                ["e_energy"] = energies.Average(),
                ["e_energy_complex"] = energiesComplex.Average(),
                ["e_energy_plus"] = energiesPlus.Average(),
                ["e_energy_complex_plus"] = energiesComplexPlus.Average(),
                ["e_energy_plus_avg"] = energiesPlusAvg.Average(),
                ["e_energy_complex_plus_avg"] = energiesComplexPlusAvg.Average(),
                ["e_distance"] = distances.Average(),
                ["e_success"] = successes.Average(),
            };
            
            foreach (var agent in ActiveAgents<AgentBasic>())
            {
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
            var data = new TrajectoryData(_timeMemory, _positionMemory, _goalPosition, _finishTime);
            var json = JsonConvert.SerializeObject(data);
            File.WriteAllText(fullSavePath, json);
        }


        private void CollectStats(Dictionary<string, float> episodeStats = null)
        {
            var distances = new List<float>();
            var speeds = new List<float>();
            var dones = new List<float>();
            var collisions = new List<float>();

            var activeSpeeds = new List<float>();
            
            foreach (var (agent, agentBasic) in ActiveAgentsTransform<AgentBasic>())
            {
                // if (agentBasic.CollectedGoal) continue;
                
                // Get distance from goal
                var agentPosition = agent.localPosition;
                var goalPosition = agentBasic.Goal.localPosition;

                var distance = Vector3.Distance(agentPosition, goalPosition);
                distances.Add(distance);
            
                // Get speed
                var speed = agent.GetComponent<Rigidbody>().velocity.magnitude;
                speeds.Add(speed);
                
                if (!agentBasic.CollectedGoal)
                {
                    activeSpeeds.Add(speed);
                }
            
                // Debug.Log($"Stats from agent {agent.name}");
                // Fraction of agents  that finished already
                dones.Add(_finished[agent] ? 1f : 0f);
                // Debug.Log(_finished[agent]);
            
                collisions.Add(agentBasic.Collision);
            }
            
            // TODO: at some point uniformize e_name and m_name

            episodeStats ??= new Dictionary<string, float>();

            episodeStats["mean_distance"] = distances.Average();
            episodeStats["mean_speed"] = speeds.Average();
            episodeStats["mean_done"] = dones.Average();
            episodeStats["mean_collision"] = collisions.Average();

            if (activeSpeeds.Count > 0)
                episodeStats["mean_active_speed"] = activeSpeeds.Average();

            
            


            // Debug.Log(collision);

            var message = MLUtils.MakeMessage(episodeStats);
            
            // var message = $"mean_dist {meanDist}\nmean_speed {meanSpeed}\nmean_finish {finished}\nmean_collision {collision}";
            statsCommunicator.StatsChannel.SendMessage(message);
            // Debug.Log("Message allegedly sent");
        }
        
        private void OnDestroy()
        {
            if (Academy.IsInitialized)
            {
                SideChannelManager.UnregisterSideChannel(StringChannel);
            }
        }

        public float[,] CompressInfo()
        {
            // Saves the global information about agents in a single array
            var array = new float[numAgents, 6];
            var agentIdx = 0;
            // var decisionTime = Time / decisionFrequency;
            foreach (var (agent, agentBasic) in ActiveAgentsTransform<AgentBasic>())
            {
                var localPosition = agent.localPosition;
                var goalPosition = agentBasic.Goal.localPosition;
                var velocity = agentBasic.Rigidbody.velocity;

                array[agentIdx, 0] = localPosition.x;
                array[agentIdx, 1] = localPosition.z;
                array[agentIdx, 2] = goalPosition.x;
                array[agentIdx, 3] = goalPosition.z;
                array[agentIdx, 4] = velocity.x;
                array[agentIdx, 5] = velocity.z;

                // _positionMemory[agentIdx, Timestep, 0] = localPosition.x;
                // _positionMemory[agentIdx, Timestep, 1] = localPosition.z;

                agentIdx++;
            }
            return array;
        }
        
        public float NormedTime => DecisionTimestep / (float) maxStep;

        IEnumerable<T> ActiveAgents<T>() where T : Component
        {
            foreach (Transform agent in transform)
            {
                if (!agent.gameObject.activeInHierarchy) continue;
                yield return agent.GetComponent<T>();
            }
        }

        IEnumerable<AgentBasic> ActiveAgents()
        {
            foreach (Transform agent in transform)
            {
                if (!agent.gameObject.activeInHierarchy) continue;
                yield return agent.GetComponent<AgentBasic>();
            }
        }
        
        IEnumerable<Transform> ActiveAgentsTransform()
        {
            foreach (Transform agent in transform)
            {
                if (!agent.gameObject.activeInHierarchy) continue;
                yield return agent;
            }
        }
        
        IEnumerable<(Transform, T)> ActiveAgentsTransform<T>() where T : Component
        {
            foreach (Transform agent in transform)
            {
                if (!agent.gameObject.activeInHierarchy) continue;
                var component = agent.GetComponent<T>();
                if (component == null) continue;
                yield return (agent, component);
            }
        }
        
        public float PhysicsDeltaTime => Time.fixedDeltaTime;
        public float DecisionDeltaTime => Time.fixedDeltaTime * decisionFrequency;
    }
}