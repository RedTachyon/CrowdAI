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
    private StatsSideChannel _statsChannel;
    private int _time;

    private bool _end = false;
    private bool _stats = false;
    
    private void Awake()
    {
        _statsChannel = new StatsSideChannel();
        SideChannelManager.RegisterSideChannel(_statsChannel);
    }

    private void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            SideChannelManager.UnregisterSideChannel(_statsChannel);
        }
    }

    public override void Initialize()
    {
        base.Initialize();
        
        Finished = new Dictionary<Transform, bool>();
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        Finished.Clear();

        _time = 0;

        foreach (Transform agent in transform)
        {
            Finished[agent] = false;
        }
        
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Collect summary statistic

        _stats = true;

        _time++;
    }

    private void CollectStats()
    {
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

        var message = $"mean_dist {meanDist}\nmean_speed {meanSpeed}\nmean_finish {finished}\nmean_collision {collision}";
        _statsChannel.SendMessage(message);
    }
    
    public void ReachGoal(Walker agent)
    {
        Finished[agent.transform] = true;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // base.Heuristic(in actionsOut);
        var contActionsOut = actionsOut.ContinuousActions;
        contActionsOut[0] = 0f;
    }

    private void LateUpdate()
    {
        if (_end)
        {
            Finish();
        }
        _end = false;

        if (_stats)
        {
            CollectStats();
        }

        _stats = false;
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
