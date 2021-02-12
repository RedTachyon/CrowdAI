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

public class AgentRandom : Walker
{


    private Material _material;
    private Color _originalColor;
    
    // public Transform goal;
    

    public override void Initialize()
    {
        base.Initialize();
        _material = GetComponent<Renderer>().material;
        _originalColor = _material.color;
        
    }

    public override void CollectObservations(VectorSensor sensor)
    {

        Collision = 0;
        
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
        // sensor.AddObservation(position.x / 10f);
        // sensor.AddObservation(position.z / 10f);
        
        // Rotation: 1
        // sensor.AddObservation(rotation.eulerAngles.y / 360f);
        
        // Goal position: 2
        // sensor.AddObservation(goalPosition.x / 10f);
        // sensor.AddObservation(goalPosition.z / 10f);
        
        // Relative position: 2
        // var relPosition = Quaternion.Inverse(rotation) * (goalPosition - position);
        var relPosition = goalPosition - position;
        sensor.AddObservation(relPosition.x / 20f);
        sensor.AddObservation(relPosition.z / 20f);
        
        // Debug.Log(relPosition);

        // var distance = (goalPosition - position).magnitude;
        // var angle = Vector3.Angle(Vector3.forward, relPosition);
        //
        // // Debug.Log($"Distance: {distance}, angle: {angle}");
        // sensor.AddObservation(distance / 20f);
        // sensor.AddObservation(angle / 180f);

        // Debug.Log(distance);
        
        // Debug.Log(relPosition);
        Debug.DrawLine(transform.position, transform.position + relPosition, Color.red, 0.02f);

        // Velocity: 2, up to ~5
        sensor.AddObservation(velocity.x / 5f);
        sensor.AddObservation(velocity.z / 5f);

        // Debug.Log(velocity);
        // sensor.AddObservation(Unfrozen);

        // REWARDS
        
        // Compute the distance-based reward
        var prevDistance = Vector3.Distance(PreviousPosition, goalPosition);
        var currentDistance = Vector3.Distance(position, goalPosition);
        // Debug.Log(currentDistance);
        // Up to ~0.1
        var diff = prevDistance - currentDistance;
        
        // Debug.Log(diff);
        
        AddReward(0.1f * diff);  // Add reward for getting closer to the goal

        // Maximum distance: 20; this puts it in the range [0, 0.1]
        // AddReward(-currentDistance / 200f);
        
        // AddReward(-0.01f);  // Small penalty at each step
        // Debug.Log($"Distance {currentDistance}");
        // Debug.Log($"Distance difference {diff}");

        PreviousPosition = position;

        _material.color = _originalColor;
        
        // Debug.Log(GetCumulativeReward());
        
        // Debug.Log($"Total reward: {GetCumulativeReward()}");

    }

    // public override void OnActionReceived(float[] vectorAction)
    // {
    //     base.OnActionReceived(vectorAction);
    //     // var angularSpeed = Unfrozen * Mathf.Clamp(vectorAction[1], -1f, 1f);
    //     // if (Mathf.Abs(angularSpeed) > 0.7f)
    //     // {
    //     //     AddReward(-0.1f * Mathf.Abs(angularSpeed));
    //     // }
    // }


    private void OnTriggerStay(Collider other)
    {
        // Debug.Log("Hitting a trigger");
        

        if (other.name == goal.name)  // Requires the goals to have unique names - not ideal, but only thing that works
        {
            AddReward(0.03f);
            GetComponentInParent<ManagerRandom>().ReachGoal(this);
            _material.color = Color.blue;
            
            // Debug.Log("Collecting a reward");
        }
    }

    protected override void OnCollisionEnter(Collision other)
    {
        base.OnCollisionEnter(other);
        
        if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
        {
            AddReward(-.01f);
            _material.color = Color.red;
            // Debug.Log($"Collision with an {other.collider.tag}!");
            // Debug.Log("I shouldn't be here");
    
        }
    }
    
    
    protected override void OnCollisionStay(Collision other)
    {
        base.OnCollisionStay(other);
        
        if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
        {
            // AddReward(-.5f);
            _material.color = Color.red;

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
