using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.MLAgents.SideChannels;
using Random = System.Random;


public class Statistician : Agent
// Abstract class that only implements the statistics collection behavior for the manager - will be useful for other scenarios
{
    protected Dictionary<Transform, bool> Finished;
    internal int Time;
    public StatsCommunicator statsCommunicator;

    private bool _stats;
    public override void Initialize()
    {
        base.Initialize();
        
        Finished = new Dictionary<Transform, bool>();
        
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        Finished.Clear();

        Time = 0;
        _stats = false;

        foreach (Transform agent in transform)
        {
            Finished[agent] = false;
        }
        
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Collect summary statistic
        
        // Debug.Log($"{name} CollectObs");

        Time++;

        sensor.AddObservation(0f);
        
        if (_stats) // Trick to avoid logging redundant stats when the episode starts?
        {
            CollectStats();
        }

        _stats = true;

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
            var goalPosition = agent.GetComponent<Walker>().goal.localPosition;

            var distance = Vector3.Distance(agentPosition, goalPosition);
            distances.Add(distance);
            
            // Get speed
            var speed = agent.GetComponent<Rigidbody>().velocity.magnitude;
            speeds.Add(speed);

            // Fraction of agents that finished already
            dones.Add(Finished[agent] ? 1f : 0f);
            // Debug.Log(_finished[agent]);
            
            collisions.Add(agent.GetComponent<Walker>().Collision);

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
    
    public void ReachGoal(Walker agent)
    {
        Finished[agent.GetComponent<Transform>()] = true;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // base.Heuristic(in actionsOut);
        var contActionsOut = actionsOut.ContinuousActions;
        contActionsOut[0] = 0f;
    }
    

    public void Finish()
    {
        foreach (Transform agent in transform)
        {
            agent.GetComponent<Agent>().EndEpisode();
        }

        EndEpisode();
    }
}
