using System;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.AI;

public enum Placement
{
    Random,
    Circle,
    Hallway,
}

public class ManagerRandom : Statistician
{
    [Range(1, 100)]
    public int numAgents = 1;
    public Placement mode;

    public Transform baseObstacle;
    public Transform obstacles;
    
    public override void Initialize()
    {
        base.Initialize();
    
        // var agent = GetComponentInChildren<AgentRandom>();
        // var goal = agent.goal;
        
        // Debug.Log("Cloning agents");
        // for (var i = 1; i < numAgents; i++)
        // {
        //     var newAgent = Instantiate(agent, transform);
        //     var newGoal = Instantiate(goal, goal.parent);
        //     
        //     newAgent.GetComponent<AgentRandom>().goal = newGoal;
        //     newAgent.name = agent.name + $" ({i})";
        //     newGoal.name = goal.name + $" ({i})";
        // }
        

    }

    public override void OnEpisodeBegin()
    {
        // Debug.Log("Manager starting an episode");
        // _done = false;
        // Debug.Log(UnityEngine.Random.state.GetHashCode());
        // UnityEngine.Random.InitState(DateTime.Now.Millisecond);
        var newMode = GetMode();
        
        numAgents = GetNumAgents();
        var currentNumAgents = transform.childCount;
        var agentsToAdd = numAgents - currentNumAgents;

        obstacles.gameObject.SetActive(newMode == Placement.Hallway);
        Debug.Log($"Number of children: {currentNumAgents}");

        // Activate the right amount of agents
        for (var i = 0; i < currentNumAgents; i++)
        {
            var active = i < numAgents;
            var currentAgent = transform.GetChild(i);
            currentAgent.gameObject.SetActive(active);
            var currentGoal = currentAgent.GetComponent<AgentRandom>().goal;
            currentGoal.gameObject.SetActive(active);
        }
        
        var baseAgent = GetComponentInChildren<AgentRandom>();
        var baseGoal = baseAgent.goal;

        // If necessary, add some more agents
        if (agentsToAdd > 0) Debug.Log($"Creating {agentsToAdd} new agents");
        
        for (var i = 0; i < agentsToAdd; i++)
        {
            var newAgent = Instantiate(baseAgent, transform);
            var newGoal = Instantiate(baseGoal, baseGoal.parent);
            
            newAgent.GetComponent<AgentRandom>().goal = newGoal;
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

            var goal = agent.GetComponent<AgentRandom>().goal;

            Vector3 newPosition;
            Vector3 goalPosition;
            Quaternion newRotation;
            
            switch (newMode)
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
        
            agent.GetComponent<AgentRandom>().PreviousPosition = agent.localPosition;
            
            // Update the counter
            agentIdx++;
        }

        // Initialize stats
        base.OnEpisodeBegin();

    }
    
    public new void ReachGoal(Walker agent)
    {
        base.ReachGoal(agent);
        // Debug.Log("I'm here!");
        // agent.goal.localPosition = new Vector3(
        //     UnityEngine.Random.Range(-9f, 9f),
        //     0.15f,
        //     UnityEngine.Random.Range(-9f, 9f));
    }

    public Placement GetMode()
    {
        var val = Academy.Instance.EnvironmentParameters.GetWithDefault("mode", -1);
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
