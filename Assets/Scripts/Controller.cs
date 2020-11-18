using System;
using System.ComponentModel;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

// Proposed reward structure:
// 16.5 total reward for approaching the goal
// 0.1 reward per decision step for reaching the goal (10 reward per 100 steps, max ~40)
// -0.01 reward per decision step for collisions (-1 reward per 100 steps)
// -0.01 reward per decision step

public class Controller : Walker
{
    // private Vector3 _startPosition;
    // private Quaternion _startRotation;

    private int _decisionPeriod;
    
    public Transform goal;

    public override void Initialize()
    {
        base.Initialize();
        _decisionPeriod = GetComponent<DecisionRequester>().DecisionPeriod;
        Debug.Log(_decisionPeriod);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // RayPerceptionSensor structure:
        // 0 - n_tags: one-hot encoding of what was hit
        // n_tags: whether *something* was hit
        // n_tags + 1: normalized distance
        
        Vector3 position = transform.localPosition;
        Vector3 rotation = transform.localRotation.eulerAngles;
        
        Vector3 velocity = Rigidbody.velocity;
        // Vector3 angularVelocity = _rigidbody.angularVelocity;
        Vector3 goalPosition = goal.localPosition;
        
        sensor.AddObservation(position.x / 20f);
        sensor.AddObservation(position.z / 20f);
        sensor.AddObservation(rotation.y / 360f);
        
        sensor.AddObservation(goalPosition.x / 20f);
        sensor.AddObservation(goalPosition.z / 20f);
        
        sensor.AddObservation(Unfrozen);

        
        // Compute the distance-based reward
        var prevDistance = Vector3.Distance(PreviousPosition, goalPosition);
        var currentDistance = Vector3.Distance(position, goalPosition);
        var diff = prevDistance - currentDistance;
        
        AddReward(1f * diff);  // Add reward for getting closer to the goal
        
        AddReward(-0.01f);  // Small penalty at each step
        // Debug.Log($"Distance {currentDistance}");
        // Debug.Log($"Distance difference {diff}");

        PreviousPosition = position;

    }
    

    private void OnTriggerStay(Collider other)
    {
        Debug.Log("Hitting a trigger");
        

        if (other.name == goal.name)  // Requires the goals to have unique names - not ideal, but only thing that works
        {
            AddReward(0.1f);
            GetComponentInParent<Manager>().ReachGoal(this);
            
            Debug.Log("Collecting a reward");
        }
    }

    private void OnCollisionStay(Collision other)
    {
        if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
        {
            AddReward(-0.01f);
            Debug.Log($"Collision with an {other.collider.tag}!");
        }
    }
    
    public Vector3 PreviousPosition { get; set; }
}
