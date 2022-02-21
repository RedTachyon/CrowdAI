using System.Collections.Generic;
using Agents;
using Initializers;
using Unity.MLAgents;
using UnityEngine;

namespace Managers
{
    public class GoalSeqManager : Manager
    {

        private List<Vector3> _obstaclePositions = new();

        public new void Awake()
        {
            base.Awake();
            // TODO: Make recursive
            
            foreach (Transform obstacle in obstacles.GetComponentsInChildren<Transform>())
            {
                if (obstacle.childCount > 0) continue;
                _obstaclePositions.Add(obstacle.position);
                // Debug.Log(obstacle.name);
            }
            
        }
        public override void ReachGoal(Agent agent)
        {
            Debug.Log("Relocating the goal");
            var goal = agent.GetComponent<AgentBasic>().goal;
            var goalPosition = MLUtils.NoncollidingPosition(
                -4f,
                4f,
                -4f,
                4f,
                goal.position.y,
                _obstaclePositions
                );
            goal.position = goalPosition;

        }
        
        public override void ResetEpisode()
        {

            Debug.Log("ResetEpisode Sequence");

            _episodeNum++;
            _initialized = true;
            mode = GetMode();
        
            numAgents = GetNumAgents();

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
        
            for (var i = 0; i < agentsToAdd; i++)
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
            
            // Find the right locations for all agents
            Debug.Log($"Total agents: {transform.childCount}");
            IInitializer initializer = Mapper.GetInitializer(mode, dataFileName);
            initializer.PlaceAgents(transform, initSize, _obstaclePositions);


            
            // Initialize stats
            _finished.Clear();

            Timestep = 0;

            foreach (Transform agent in transform)
            {
                _finished[agent] = false;
                agent.GetComponent<AgentBasic>().OnEpisodeBegin();
            }

        }
    }
}