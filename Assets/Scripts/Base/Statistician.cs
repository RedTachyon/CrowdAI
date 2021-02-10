using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.MLAgents.SideChannels;


public class Statistician : Agent
// Abstract class that only implements the statistics collection behavior for the manager - will be useful for other scenarios
{
    protected Dictionary<Transform, bool> Finished;

    public override void Initialize()
    {
        base.Initialize();
        
        Finished = new Dictionary<Transform, bool>();

        foreach (Transform agent in transform)
        {
            Walker walker = agent.GetComponent<Walker>();
            walker.startPosition = agent.localPosition;
        }
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        Finished.Clear();

        foreach (Transform agent in transform)
        {
            Finished[agent] = false;
        }
        
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Collect summary statistics
        var distances = new List<float>();
        var speeds = new List<float>();
        var dones = new List<float>();
        var collisions = new List<int>();
        
        foreach (Transform agent in transform)
        {
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

        sensor.AddObservation(meanDist);
        sensor.AddObservation(meanSpeed);
        sensor.AddObservation(finished);
        sensor.AddObservation(collision);
        
        // Debug.Log(finished);
    }
    
    public void ReachGoal(Walker agent)
    {
        Finished[agent.transform] = true;
        // agent.EndEpisode();
        // agent.Freeze();

        if (!Finished.Values.Contains(false))
        {
            
            // _done = true;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // base.Heuristic(in actionsOut);
        var contActionsOut = actionsOut.ContinuousActions;
        contActionsOut[0] = 0f;
    }
}
