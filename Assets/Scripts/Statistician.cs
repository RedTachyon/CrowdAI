using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.MLAgents.SideChannels;


public class Statistician : Agent
// Abstract class that only implements the statistics collection behavior for the manager - will be useful for other scenarios
{
    protected Dictionary<Transform, bool> _finished;


    public override void CollectObservations(VectorSensor sensor)
    {
        // Collect summary statistics
        var distances = new List<float>();
        var speeds = new List<float>();
        var dones = new List<float>();
        
        foreach (Transform agent in transform)
        {
            // Get distance from goal
            var agentPosition = agent.position;
            var goalPosition = agent.GetComponent<AgentController>().goal.position;

            var distance = Vector3.Distance(agentPosition, goalPosition);
            distances.Add(distance);
            
            // Get speed
            var speed = agent.GetComponent<Rigidbody>().velocity.magnitude;
            speeds.Add(speed);

            // Fraction of agents that finished already
            dones.Add(_finished[agent] ? 1f : 0f);
        }

        var meanDist = distances.Average();
        var meanSpeed = speeds.Average();
        var finished = dones.Average();

        sensor.AddObservation(meanDist);
        sensor.AddObservation(meanSpeed);
        sensor.AddObservation(finished);
        
        // Debug.Log(finished);
    }
    
}
