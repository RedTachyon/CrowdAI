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

namespace Managers
{

    public class Manager : MonoBehaviour
    {
        [Range(1, 100)]
        public int numAgents = 1;
        public InitializerEnum mode;
        public string dataFileName;

        [Range(1, 1000)]
        public int maxStep = 500;

        [Range(1, 10)] public int decisionFrequency = 1;

        private Dictionary<Transform, bool> _finished;
        internal int Timestep;
        public StatsCommunicator statsCommunicator;

        public StringChannel StringChannel;

        public Transform obstacles;

        private SimpleMultiAgentGroup _agentGroup;

        private bool _initialized;
        private int _episodeNum;

        private float[,,] _positionMemory;
        private float[] _timeMemory;

        private static Manager _instance;
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
            _agentGroup = new SimpleMultiAgentGroup();

            foreach (Transform agent in transform)
            {
                _agentGroup.RegisterAgent(agent.GetComponent<Agent>());
            }
            
            StringChannel = new StringChannel();
            SideChannelManager.RegisterSideChannel(StringChannel);

            _episodeNum = 0;
            
            // Debug.Log(Quaternion.AngleAxis(90, Vector3.up));
            // Debug.Log(MLUtils.SquashUniform(new Vector2(0f ,1f)));
            // Debug.Log(MLUtils.SquashUniform(new Vector2(1f ,0f)));
            //
            // Debug.Log(MLUtils.SquashUniform(new Vector2(Mathf.Sqrt(2)/2 ,Mathf.Sqrt(2)/2)));

        }

        public void ResetEpisode()
        {

            Debug.Log("ResetEpisode");

            _episodeNum++;
            _initialized = true;
            mode = GetMode();
        
            numAgents = GetNumAgents();

            _positionMemory = new float[numAgents, maxStep * decisionFrequency, 2];
            _timeMemory = new float[maxStep * decisionFrequency];

            var currentNumAgents = transform.childCount;
            var agentsToAdd = numAgents - currentNumAgents;

            obstacles.gameObject.SetActive(mode == InitializerEnum.Hallway);
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

                // TODO: this will crash?
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
        
            for (var i = 0; i < agentsToAdd; i++)
            {
                var newAgent = Instantiate(baseAgent, transform);
                var newGoal = Instantiate(baseGoal, baseGoal.parent);
            
                newAgent.GetComponent<AgentBasic>().goal = newGoal;
                newAgent.name = baseAgent.name + $" ({i})";
                newGoal.name = baseGoal.name + $" ({i})";
            }
        
            
            
            // Find the right locations for all agents
            Debug.Log($"Total agents: {transform.childCount}");
            IInitializer initializer = Mapper.GetInitializer(mode, dataFileName);
            initializer.PlaceAgents(transform);

            // Initialize stats
            _finished.Clear();

            Timestep = 0;

            foreach (Transform agent in transform)
            {
                _finished[agent] = false;
            }

        }
        public void ReachGoal(Agent agent)
        {
            _finished[agent.GetComponent<Transform>()] = true;
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
            
            if (Timestep >= maxStep * decisionFrequency)
            {
                
                if (Params.SavePath != "") WriteTrajectory();
                else Debug.Log("Oops, not saving anything");

                Debug.Log("Resetting");
                
                _agentGroup.EndGroupEpisode();
                ResetEpisode();
            }

            ///////////////////////
            // Log the positions //
            ///////////////////////
            
            var agentIdx = 0;
            // var decisionTime = Time / decisionFrequency;
            foreach (Transform agent in transform)
            {
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


    
            CollectStats();


        }
    
    
        private void CollectStats()
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
        
            // Debug.Log(collision);

        
            var message = $"mean_dist {meanDist}\nmean_speed {meanSpeed}\nmean_finish {finished}\nmean_collision {collision}";
            statsCommunicator.StatsChannel.SendMessage(message);
            // Debug.Log("Message allegedly sent");
        }

        public InitializerEnum GetMode()
        {
            var val = Academy.Instance.EnvironmentParameters.GetWithDefault("mode", -1f);
            InitializerEnum currentMode;
            if (val < -0.5f) // == -1f 
            {
                currentMode = mode;
            }
            else if (val < 0.5f) // == 0f
            {
                currentMode = InitializerEnum.Random;
            } 
            else if (val < 1.5f) // == 1f
            {
                currentMode = InitializerEnum.Circle;
            }
            else if (val < 2.5f) // == 2f
            {
                currentMode = InitializerEnum.Hallway;
            }
            else
            {
                currentMode = InitializerEnum.JsonInitializer;
            }

            return currentMode;
        }

        public int GetNumAgents()
        {
            var val = Academy.Instance.EnvironmentParameters.GetWithDefault("agents", -1f);
            int agents;
            agents = val < 0 ? numAgents : Mathf.RoundToInt(val);

            return agents;
        }

        private void OnDestroy()
        {
            if (Academy.IsInitialized)
            {
                SideChannelManager.UnregisterSideChannel(StringChannel);
            }
        }
    }
}