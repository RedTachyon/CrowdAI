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

public class AgentController : Walker
{
    // private Vector3 _startPosition;
    // private Quaternion _startRotation;

    private int _decisionPeriod;
    private Material _material;
    private Color _originalColor;
    
    // public Transform goal;
    

    public override void Initialize()
    {
        base.Initialize();
        _decisionPeriod = GetComponent<DecisionRequester>().DecisionPeriod;
        _material = GetComponent<Renderer>().material;
        _originalColor = _material.color;
        
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // RayPerceptionSensor structure:
        // 0 - n_tags: one-hot encoding of what was hit
        // n_tags: whether *something* was hit
        // n_tags + 1: normalized distance
        
        Vector3 position = transform.localPosition;
        Quaternion rotation = transform.localRotation;
        
        Vector3 velocity = Rigidbody.velocity;
        // Vector3 angularVelocity = _rigidbody.angularVelocity;
        Vector3 goalPosition = goal.localPosition;
        
        // Position: 2
        sensor.AddObservation(position.x / 10f);
        sensor.AddObservation(position.z / 10f);
        
        // Rotation: 1
        sensor.AddObservation(rotation.eulerAngles.y / 360f);
        
        // Goal position: 2
        sensor.AddObservation(goalPosition.x / 10f);
        sensor.AddObservation(goalPosition.z / 10f);
        
        // Relative position: 2
        var relPosition = Quaternion.Inverse(rotation) * (goalPosition - position);
        sensor.AddObservation(relPosition.x / 10f);
        sensor.AddObservation(relPosition.z / 10f);
        
        Debug.Log(relPosition);
        Debug.DrawLine(transform.position, transform.position + rotation * relPosition, Color.red, 0.02f);

        // Velocity: 2
        sensor.AddObservation(velocity.x / 3f);
        sensor.AddObservation(velocity.z / 3f);
        
        sensor.AddObservation(Unfrozen);

        // REWARDS
        
        // Compute the distance-based reward - temporarily (?) deprecated
        // var prevDistance = Vector3.Distance(PreviousPosition, goalPosition);
        var currentDistance = Vector3.Distance(position, goalPosition);
        // var diff = prevDistance - currentDistance;
        //
        // AddReward(1f * diff);  // Add reward for getting closer to the goal

        // Maximum distance: 20; this puts it in the range [0, 0.1]
        AddReward(-currentDistance / 200f);
        
        // AddReward(-0.01f);  // Small penalty at each step
        // Debug.Log($"Distance {currentDistance}");
        // Debug.Log($"Distance difference {diff}");

        PreviousPosition = position;

        _material.color = _originalColor;

    }
    

    private void OnTriggerStay(Collider other)
    {
        // Debug.Log("Hitting a trigger");
        

        if (other.name == goal.name)  // Requires the goals to have unique names - not ideal, but only thing that works
        {
            AddReward(0.1f / _decisionPeriod);
            GetComponentInParent<Statistician>().ReachGoal(this);
            _material.color = Color.blue;
            
            // Debug.Log("Collecting a reward");
        }
    }

    private void OnCollisionStay(Collision other)
    {
        if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
        {
            AddReward(-0.1f / _decisionPeriod);
            _material.color = Color.red;
            // Debug.Log($"Collision with an {other.collider.tag}!");

        }
    }

    // private void OnCollisionExit(Collision other)
    // {
    //     _material.color = _originalColor;
    // }
    //
    // private void OnCollisionEnter(Collision other)
    // {
    //     if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
    //     {
    //         _material.color = Color.red;
    //     }
    // }

    public Vector3 PreviousPosition { get; set; }
}
