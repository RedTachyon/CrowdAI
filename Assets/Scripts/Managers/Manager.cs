using System;
using System.Collections.Generic;
using System.Linq;
using Agents;
using Initializers;
using Unity.MLAgents;
using UnityEngine;

namespace Managers
{
    public enum Placement
    {
        Random,
        Circle,
        Hallway,
    }

    public class Manager : MonoBehaviour
    {
        [Range(1, 100)]
        public int numAgents = 1;
        public InitializerEnum mode;
    
        [Range(1, 1000)]
        public int maxStep = 500;

        [Range(1, 10)] public int decisionFrequency = 1;

        private Dictionary<Transform, bool> _finished;
        internal int Time;
        public StatsCommunicator statsCommunicator;

        public Transform obstacles;

        private SimpleMultiAgentGroup _agentGroup;

        private bool _initialized;

        private float[,,] _positionMemory;


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
        }

        public void ResetEpisode()
        {

            Debug.Log("ResetEpisode");

            
            _initialized = true;
            mode = GetMode();
        
            numAgents = GetNumAgents();
            _positionMemory = new float[numAgents,maxStep,2];

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
            IInitializer initializer = Mapper.GetInitializer(mode);
            initializer.PlaceAgents(transform);

            // Initialize stats
            _finished.Clear();

            Time = 0;

            foreach (Transform agent in transform)
            {
                _finished[agent] = false;
            }

        }
        public void ReachGoal(Agent agent)
        {
            _finished[agent.GetComponent<Transform>()] = true;
        }

        private void FixedUpdate()
        {
            if (!_initialized) return;
            
            if (Time >= maxStep * decisionFrequency)
            {
                Debug.Log("Resetting");
                _agentGroup.EndGroupEpisode();
                ResetEpisode();
            }

            // Log the positions
            
            if (Time % decisionFrequency == 0)
            {
                var agentIdx = 0;
                var decisionTime = Time / decisionFrequency;
                foreach (Transform agent in transform)
                {
                    var localPosition = agent.localPosition;
                    _positionMemory[agentIdx, decisionTime, 0] = localPosition.x;
                    _positionMemory[agentIdx, decisionTime, 1] = localPosition.z;

                    agentIdx++;
                }
            }
            
            foreach (Transform agent in transform)
            {
                if (Time % decisionFrequency == 0)
                {
                    Debug.Log($"Action time: {Time}");
                    agent.GetComponent<Agent>().RequestDecision();
                }
                else
                {
                    agent.GetComponent<Agent>().RequestAction();
                }
            }
            Time++;
        
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
            else // == 2f
            {
                currentMode = InitializerEnum.Hallway;
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
    
    }
}