using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Net;
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
    private bool _collectedGoal;

    // public CBufferSensorComponent bufferSensor;
    private BufferSensorComponent _bufferSensor;
    
    // public Transform goal;


    public override void Initialize()
    {
        base.Initialize();
        _material = GetComponent<Renderer>().material;
        _originalColor = _material.color;
        _bufferSensor = GetComponent<BufferSensorComponent>();

    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        _collectedGoal = false;

    }


    public override void CollectObservations(VectorSensor sensor)
    {
        base.CollectObservations(sensor);
        // Debug.Log($"{name} CollectObs at step {GetComponentInParent<Statistician>().Time}");
        
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
        var relPosition = Quaternion.Inverse(rotation) * (goalPosition - position);
        // var relPosition = goalPosition - position;
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

        // Velocity: 2, up to ~5
        sensor.AddObservation(velocity.x / 5f);
        sensor.AddObservation(velocity.z / 5f);

        // Debug.Log(velocity);
        // sensor.AddObservation(Unfrozen);

        // REWARDS
        
        // Compute the distance-based reward
        var prevDistance = Vector3.Distance(PreviousPosition, goalPosition);
        var currentDistance = Vector3.Distance(position, goalPosition);
        // Up to ~0.1
        var diff = prevDistance - currentDistance;

        // Debug.Log($"Distance {currentDistance}");
        // Debug.Log($"Distance difference {diff}");


        
        // Compute the reward
        AddReward(Params.Potential * diff);  // Add reward for getting closer to the goal

        var layerMask = 1 << 3;
        var nearbyObjects =
            Physics.OverlapSphere(transform.position, Params.SightRadius, layerMask)
                .Where(c => c.CompareTag("Agent") & c.transform != transform) // Get only agents 
                .OrderBy(c => Vector3.Distance(c.transform.localPosition, transform.localPosition))
                .Select(c => MLUtils.GetColliderInfo(transform, c))
                .Take(Params.SightAgents);
        
        // Debug.Log(nearbyObjects);
        foreach (var agentInfo in nearbyObjects)
        {
            Debug.Log(String.Join(",", agentInfo));
            _bufferSensor.AppendObservation(agentInfo);
        }


        // Debug.Log($"Total reward: {GetCumulativeReward()}");

        // Debug graphics
        Debug.DrawLine(transform.position, goal.position, Color.red, Time.fixedDeltaTime);
        // Debug.Log($"Current position: {transform.position}. Previous position: {PreviousPosition}");
        var parentPosition = transform.parent.position;
        var absPrevPosition = PreviousPosition + parentPosition;
        Debug.DrawLine(transform.position, absPrevPosition, Color.green, 20*Time.fixedDeltaTime);


        
        // Final updates
        PreviousPosition = transform.localPosition;

        _material.color = _originalColor;

    }


    private void OnTriggerStay(Collider other)
    {
        // Debug.Log("Hitting a trigger");
        
        if (other.name != goal.name) return;
        
        // Give the goal reward at most only once per episode
        if (!_collectedGoal)
        {
            AddReward(Params.Goal);
        }
        _collectedGoal = true;

        GetComponentInParent<ManagerRandom>().ReachGoal(this);
        _material.color = Color.blue;
            
        // Debug.Log("Collecting a reward");
    }

    protected override void OnCollisionEnter(Collision other)
    {
        base.OnCollisionEnter(other);
        
        if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
        {
            AddReward(Params.Collision);
            _material.color = Color.red;
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

    public Vector3 PreviousPosition { get; set; }
}
