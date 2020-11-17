using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;


// TODO: figure out the side channels for extra control?
public class Controller : Agent
{
    private Rigidbody _rigidbody;
    private Vector3 _previousPosition;
    // private Vector3 _startPosition;
    // private Quaternion _startRotation;

    public float moveSpeed = 50f;
    public float rotationSpeed = 3f;

    public float dragFactor = 5f;
    
    private int _unfrozen = 1;

    public Transform goal;

    public override void Initialize()
    {
        _rigidbody = GetComponent<Rigidbody>();
        // _startPosition = transform.localPosition;
        // _startRotation = transform.localRotation;
    }

    public override void OnEpisodeBegin()
    {
        // Unfreeze();
        // Vector3 startPos = new Vector3(-9f, 0.25f, UnityEngine.Random.Range(-9f, 9f));
        // Vector3 startPos = _startPosition;
        // transform.localPosition = startPos;
        // transform.localRotation = _startRotation;
        // // transform.localRotation = Quaternion.Euler(0f, 90f, 0f);
        //
        // _rigidbody.velocity = Vector3.zero;
        // _rigidbody.angularVelocity = Vector3.zero;
        //
        // _previousPosition = startPos;


    }

    public override void OnActionReceived(float[] vectorAction)
    {
        // TODO: consider changing the environment dynamics
        
        // Forward velocity
        var linearSpeed = _unfrozen * Mathf.Clamp(vectorAction[0], -0.3f, 1.0f);
        
        // Angular velocity
        var angularSpeed = _unfrozen * Mathf.Clamp(vectorAction[1], -1f, 1f);
        
        // Apply the force
        Vector3 force = transform.forward * linearSpeed * moveSpeed;

        // Reduce the velocity friction-like
        // Vector3 drag = -dragFactor * _rigidbody.velocity.magnitude * _rigidbody.velocity.normalized;
        Vector3 drag = -dragFactor * _rigidbody.velocity;
        // _rigidbody.AddForce(force + drag);

        // Apply the rotation
        Vector3 rotation = transform.rotation.eulerAngles + Vector3.up * angularSpeed * rotationSpeed;
        // _rigidbody.MoveRotation(Quaternion.Euler(rotation));

        _rigidbody.rotation = Quaternion.Euler(rotation);
    }

    public override void Heuristic(float[] actionsOut)
    {
        var forwardValue = 0f;
        var rotationValue = 0f;
        
        if (Input.GetKey(KeyCode.W)) forwardValue = 1f;
        if (Input.GetKey(KeyCode.S)) forwardValue = -1f;
        
        if (Input.GetKey(KeyCode.D)) rotationValue = 1f;
        if (Input.GetKey(KeyCode.A)) rotationValue = -1f;
        

        
        actionsOut[0] = forwardValue;
        actionsOut[1] = rotationValue;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // RayPerceptionSensor structure:
        // 0 - n_tags: one-hot encoding of what was hit
        // n_tags: whether *something* was hit
        // n_tags + 1: normalized distance
        
        Vector3 position = transform.localPosition;
        Vector3 rotation = transform.localRotation.eulerAngles;
        
        Vector3 velocity = _rigidbody.velocity;
        // Vector3 angularVelocity = _rigidbody.angularVelocity;
        Vector3 goalPosition = goal.localPosition;
        
        sensor.AddObservation(position.x / 20f);
        sensor.AddObservation(position.z / 20f);
        sensor.AddObservation(rotation.y / 360f);
        
        sensor.AddObservation(goalPosition.x / 20f);
        sensor.AddObservation(goalPosition.z / 20f);
        
        sensor.AddObservation(_unfrozen);

        
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


    // private void OnTriggerEnter(Collider other)
    // {
    //     if (other == goal.GetComponent<Collider>())
    //     {
    //         AddReward(2f);
    //         // Debug.Log("Got the goal!");
    //         GetComponentInParent<Manager>().ReachGoal(this);
    //     }
    // }

    private void OnTriggerStay(Collider other)
    {
        if (other == goal.GetComponent<Collider>())
        {
            AddReward(0.1f);
            GetComponentInParent<Manager>().ReachGoal(this);
        }
    }

    private void OnCollisionStay(Collision other)
    {
        if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
        {
            AddReward(-0.01f);
            // Debug.Log($"Collision with an {other.collider.tag}!");
        }
    }
    
    public void Freeze()
    {
        _unfrozen = 0;
        _rigidbody.constraints = _rigidbody.constraints |
                                 RigidbodyConstraints.FreezePositionX | 
                                 RigidbodyConstraints.FreezePositionZ | 
                                 RigidbodyConstraints.FreezeRotationY;
        
        Debug.Log("Freezing agent");
        
        GetComponent<Controller>().enabled = false;
        // GetComponent<DecisionRequester>().enabled = false;
        // GetComponent<DecisionRequester>().DecisionPeriod = Int32.MaxValue;
        
    }

    public void Unfreeze()
    {
        GetComponent<Controller>().enabled = true;
        // GetComponent<DecisionRequester>().enabled = true;
        // GetComponent<DecisionRequester>().DecisionPeriod = 5;

        Debug.Log("Unfreezing agent");

        
        _unfrozen = 1;
        _rigidbody.constraints &= ~RigidbodyConstraints.FreezePositionX;
        _rigidbody.constraints &= ~RigidbodyConstraints.FreezePositionZ;
        _rigidbody.constraints &= ~RigidbodyConstraints.FreezeRotationY;
    }

    public Vector3 PreviousPosition
    {
        get => _previousPosition;
        set => _previousPosition = value;
    }
}
