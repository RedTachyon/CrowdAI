using System;
using System.Collections.Specialized;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.Serialization;


public class Walker : Agent
{
    // A basic agent that only implements the environment dynamics, i.e. walking around with friction
    protected Rigidbody Rigidbody;

    public float moveSpeed = 25f;
    public float rotationSpeed = 3f;

    public float dragFactor = 5f;

    protected int Unfrozen = 1;

    internal int Collision = 0;


    public Transform goal;


    [HideInInspector] public Vector3 startPosition;

    [HideInInspector] public Quaternion startRotation;


    public override void Initialize()
    {
        Rigidbody = GetComponent<Rigidbody>();
        // startY = transform.localPosition.y;
        startPosition = transform.localPosition;
        startRotation = transform.localRotation;

        
    }


    public override void OnActionReceived(ActionBuffers actions)
    {
        base.OnActionReceived(actions);
        // Debug.Log($"{name} OnAction at step {GetComponentInParent<Statistician>().Time}");
        
        Collision = 0;
        
        var vectorAction = actions.ContinuousActions;

        // Forward velocity
        var xSpeed = Unfrozen * Mathf.Clamp(vectorAction[0], -1f, 1f);
        
        // Angular velocity
        var zSpeed = Unfrozen * Mathf.Clamp(vectorAction[1], -1f, 1f);
        
        var velocity = Rigidbody.velocity;
        
        // Debug.Log(velocity);

        // Apply the force
        // Vector3 force = transform.forward * linearSpeed * moveSpeed;
        // Apply the rotation
        // Vector3 rotation = transform.rotation.eulerAngles + Vector3.up * angularSpeed * rotationSpeed;
        // Rigidbody.rotation = Quaternion.Euler(rotation);
        
        Vector3 force = new Vector3(xSpeed, 0f, zSpeed).normalized * moveSpeed;
        
        
        // Reduce the velocity friction-like
        Vector3 drag = -dragFactor * velocity;
        Rigidbody.AddForce(force + drag);

        // Rigidbody.velocity = force / 10f;

        var rotation = Rigidbody.rotation;
        var forward = rotation * Vector3.forward;

        var dirVector = force;
        
        if (dirVector.magnitude > .1f)
        {
            var orthogonal = Vector3.Cross(Vector3.up, forward).normalized;
            var angle = Vector3.Angle(forward, dirVector.normalized) / 180f;
            // var dot = Vector3.Dot(rotation * Vector3.forward, dirVector.normalized);
            var sign = Mathf.Sign(Vector3.Dot(orthogonal, dirVector));

            // Debug.Log(Vector3.SignedAngle(rotation * Vector3.forward, dirVector.normalized, Vector3.up));


            var direction = Vector3.MoveTowards(
                forward, 
                sign * orthogonal, 
                Mathf.Min(0.5f * angle, 0.2f)
            );
            Rigidbody.rotation = Quaternion.LookRotation(direction);
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // base.Heuristic(in actionsOut);

        var cActionsOut = actionsOut.ContinuousActions;

        var xValue = 0f;
        var zValue = 0f;
        Vector3 force;

        // Only for polar WASD controls
        // Ratio allows the agent to turn more or less in place, but still turn normally while moving.
        // The higher the ratio, the smaller circle the agent makes while turning in place (A/D)
        const float ratio = 1f;
        
        if (Input.GetKey(KeyCode.W)) zValue = 1f;
        if (Input.GetKey(KeyCode.S)) zValue = -1f;
        
        if (Input.GetKey(KeyCode.D)) xValue = 1f/ratio;
        if (Input.GetKey(KeyCode.A)) xValue = -1f/ratio;

        if (Input.anyKey)
        {
            force = new Vector3(xValue, 0, zValue);
        }
        else
        {
            force = goal.position - transform.position;
            force.y = 0f;
            force = force.normalized;
        }

        // if ((Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.S))
        //     &&
        //     (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.D)))
        // {
        //     xValue *= ratio;
        // }
        //
        // force = transform.rotation * force;
        //
        // Debug.Log(force);
        
        cActionsOut[0] = force.x;
        cActionsOut[1] = force.z;
    }

    protected virtual void OnCollisionEnter(Collision other)
    {
        // Debug.Log($"{name} in collision at step {GetComponentInParent<Statistician>().Time}");
        if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
        {
            Collision = 1;
        }
    }

    protected virtual void OnCollisionStay(Collision other)
    {
        // Debug.Log($"{name} in collision at step {GetComponentInParent<Statistician>().Time}");
        if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
        {
            Collision = 1;
        }
    }

    // public void Freeze()
    // {
    //     Unfrozen = 0;
    //     Rigidbody.constraints = Rigidbody.constraints |
    //                             RigidbodyConstraints.FreezePositionX | 
    //                             RigidbodyConstraints.FreezePositionZ | 
    //                             RigidbodyConstraints.FreezeRotationY;
    //     
    //     Debug.Log("Freezing agent");
    //
    //     enabled = false;
    //     // GetComponent<DecisionRequester>().enabled = false;
    //     // GetComponent<DecisionRequester>().DecisionPeriod = Int32.MaxValue;
    //
    // }
    //
    // public void Unfreeze()
    // {
    //     enabled = true;
    //     // GetComponent<DecisionRequester>().enabled = true;
    //     // GetComponent<DecisionRequester>().DecisionPeriod = 5;
    //
    //     Debug.Log("Unfreezing agent");
    //
    //     
    //     Unfrozen = 1;
    //     Rigidbody.constraints &= ~RigidbodyConstraints.FreezePositionX;
    //     Rigidbody.constraints &= ~RigidbodyConstraints.FreezePositionZ;
    //     Rigidbody.constraints &= ~RigidbodyConstraints.FreezeRotationY;
    // }

}
