using System;
using System.Collections.Generic;
using System.Linq;
using Agents;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Managers
{

    public class AnimalManager : MonoBehaviour
    {
        [Range(1, 100)]
        public int numAgents = 1;
    
        [Range(1, 1000)]
        public int maxStep = 500;

        private int _time;
        public StatsCommunicator statsCommunicator;
        
        private SimpleMultiAgentGroup _agentGroup;
    
        public void Awake()
        {
            Academy.Instance.OnEnvironmentReset += ResetEpisode;
            _agentGroup = new SimpleMultiAgentGroup();

            foreach (Transform agent in transform)
            {
                _agentGroup.RegisterAgent(agent.GetComponent<Agent>());
            }
            
        }

        public void ResetEpisode()
        {
            
            numAgents = GetNumAgents();
            var currentNumAgents = transform.childCount;
            var agentsToAdd = numAgents - currentNumAgents;

            Debug.Log($"Number of children: {currentNumAgents}");

            // Activate the right amount of agents
            for (var i = 0; i < currentNumAgents; i++)
            {
                var active = i < numAgents;
                var currentAgent = transform.GetChild(i);
                currentAgent.gameObject.SetActive(active);

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
        
            var baseAgent = GetComponentInChildren<Animal>();

            // If necessary, add some more agents
            if (agentsToAdd > 0) Debug.Log($"Creating {agentsToAdd} new agents");
        
            for (var i = 0; i < agentsToAdd; i++)
            {
                var newAgent = Instantiate(baseAgent, transform);
                newAgent.name = baseAgent.name + $" ({i})";
            }
        
            // Find the right locations for all agents
        
            Debug.Log($"Total agents: {transform.childCount}");

            var agentIdx = 0;
            var placedAgents = new List<Vector3>();

            foreach (Transform agent in transform)
            {
                if (!agent.gameObject.activeSelf) continue;
                
                Vector3 newPosition;
                Quaternion newRotation;

                        // Choose a new location for the agent and the goal
                newPosition = MLUtils.NoncollidingPosition(
                    -9f,
                    9f,
                    -9f,
                    9f,
                    agent.localPosition.y,
                    placedAgents);

                newRotation = Quaternion.Euler(0f, UnityEngine.Random.Range(0f, 360f), 0f);


                // Set the new positions
                agent.localPosition = newPosition;
                agent.localRotation = newRotation;

            
                // Save the placed agents
                placedAgents.Add(newPosition);

                // Reset the dynamics
                agent.GetComponent<Rigidbody>().velocity = Vector3.zero;
                agent.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        
                // agent.GetComponent<Animal>().PreviousPosition = agent.localPosition;
            
                // Update the counter
                agentIdx++;
            }

            // Initialize stats

            _time = 0;
            

        }

        private void FixedUpdate()
        {
            foreach (Transform agent in transform)
            {
                agent.GetComponent<Agent>().RequestDecision();
            }
            _time++;
        
            // Debug.Log(_time);

            if (_time > maxStep)
            {
                Debug.Log("Resetting");
                _agentGroup.EndGroupEpisode();
                ResetEpisode();
            }
    
            CollectStats();


        }
    
    
        private void CollectStats()
        {
            var speeds = new List<float>();
            var collisions = new List<int>();
        
            foreach (Transform agent in transform)
            {
                if (!agent.gameObject.activeSelf) continue;
                // Get distance from goal
                var agentPosition = agent.localPosition;

            
                // Get speed
                var speed = agent.GetComponent<Rigidbody>().velocity.magnitude;
                speeds.Add(speed);
            
                // Debug.Log($"Stats from agent {agent.name}");
                // Fraction of agents that finished already
            

            }
            var meanSpeed = speeds.Average();
        
            // Debug.Log(collision);


            var message = $"mean_speed {meanSpeed}";

            statsCommunicator.StatsChannel.SendMessage(message);
            // Debug.Log("Message allegedly sent");
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