using System;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.Serialization;

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
    public Placement mode;
    
    [FormerlySerializedAs("MaxStep")] [Range(1, 1000)]
    public int maxStep = 500;

    private Dictionary<Transform, bool> _finished;
    internal int Time;
    public StatsCommunicator statsCommunicator;

    public Transform obstacles;

    private SimpleMultiAgentGroup _agentGroup;
    
    public void Awake()
    {
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

        mode = GetMode();
        
        numAgents = GetNumAgents();
        var currentNumAgents = transform.childCount;
        var agentsToAdd = numAgents - currentNumAgents;

        obstacles.gameObject.SetActive(mode == Placement.Hallway);
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

        var agentIdx = 0;
        var placedAgents = new List<Vector3>();
        var placedGoals = new List<Vector3>();

        foreach (Transform agent in transform)
        {
            if (!agent.gameObject.activeSelf) continue;

            var goal = agent.GetComponent<AgentBasic>().goal;

            Vector3 newPosition;
            Vector3 goalPosition;
            Quaternion newRotation;
            
            switch (mode)
            {
                case Placement.Random:
                {
                    // Choose a new location for the agent and the goal
                    newPosition = MLUtils.NoncollidingPosition(
                        -9f,
                        9f,
                        -9f,
                        9f,
                        agent.localPosition.y,
                        placedAgents);

                    goalPosition = MLUtils.NoncollidingPosition(
                        -9f,
                        9f,
                        -9f,
                        9f,
                        goal.localPosition.y,
                        placedGoals);

                    newRotation = Quaternion.Euler(0f, UnityEngine.Random.Range(0f, 360f), 0f);
                    break;
                }
                case Placement.Circle:
                {
                    // Place agents in a centered circle of radius 9, with goals on the opposite side
                    const float r = 9;
                    var x = r * Mathf.Cos((float) agentIdx / numAgents * Constants.Tau);
                    var z = r * Mathf.Sin((float) agentIdx / numAgents * Constants.Tau);
                    newPosition = new Vector3(x, agent.localPosition.y, z);
                    goalPosition = new Vector3(-x, goal.localPosition.y, -z);
                    newRotation = Quaternion.LookRotation(goalPosition, Vector3.up);

                    // Debug.Log($"Placing an agent at x={x}, z={z}");
                    break;
                }
                case Placement.Hallway:
                {
                    // Place half the agents in one corridor, and half in the other
                    if (agentIdx < numAgents / 2)
                    {
                        newPosition = MLUtils.NoncollidingPosition(
                            6f,
                            9f,
                            -4f,
                            4f,
                            agent.localPosition.y,
                            placedAgents
                        );

                    }
                    else
                    {
                        newPosition = MLUtils.NoncollidingPosition(
                            -4f,
                            4f,
                            6f,
                            9f,
                            agent.localPosition.y,
                            placedAgents
                        );
                        
                    }
                    
                    
                    goalPosition = Quaternion.AngleAxis(180, Vector3.up) * newPosition;
                    goalPosition.y = goal.localPosition.y;
                    newRotation = Quaternion.LookRotation(goalPosition, Vector3.up);
                    break;
                }
                default:
                {
                    throw new ArgumentOutOfRangeException();
                }
                
            }
            
            // Set the new positions
            agent.localPosition = newPosition;
            agent.localRotation = newRotation;

            goal.localPosition = goalPosition;
            
            // Save the placed agents
            placedAgents.Add(newPosition);
            placedGoals.Add(goalPosition);

            // Reset the dynamics
            agent.GetComponent<Rigidbody>().velocity = Vector3.zero;
            agent.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        
            agent.GetComponent<AgentBasic>().PreviousPosition = agent.localPosition;
            
            // Update the counter
            agentIdx++;
        }

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
        foreach (Transform agent in transform)
        {
            agent.GetComponent<Agent>().RequestDecision();
        }
        Time++;
        
        Debug.Log(Time);

        if (Time > maxStep)
        {
            Debug.Log("Resetting");
            _agentGroup.EndGroupEpisode();
            ResetEpisode();
        }
    
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

            // Fraction of agents that finished already
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

    public Placement GetMode()
    {
        var val = Academy.Instance.EnvironmentParameters.GetWithDefault("mode", -1f);
        Placement currentMode;
        if (val < -0.5f) // == -1f 
        {
            currentMode = mode;
        }
        else if (val < 0.5f) // == 0f
        {
            currentMode = Placement.Random;
        } 
        else if (val < 1.5f) // == 1f
        {
            currentMode = Placement.Circle;
        }
        else // == 2f
        {
            currentMode = Placement.Hallway;
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
