using System;
using System.Collections.Specialized;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;


public class Walker : Agent
{
    // A basic agent that only implements the environment dynamics, i.e. walking around with friction
    protected Rigidbody Rigidbody;

    public float moveSpeed = 25f;
    public float rotationSpeed = 3f;

    public float dragFactor = 5f;

    protected int Unfrozen = 1;

    public Transform goal;

    [HideInInspector]
    public float StartY;


    public override void Initialize()
    {
        Rigidbody = GetComponent<Rigidbody>();
        StartY = transform.localPosition.y;
    }
    

    public override void OnActionReceived(float[] vectorAction)
    {
        
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
        
        Vector3 force = new Vector3(xSpeed, 0f, zSpeed) * moveSpeed;
        
        
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
            var angle = Vector3.Angle(forward, dirVector.normalized)/180f;
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

    public override void Heuristic(float[] actionsOut)
    {
        var xValue = 0f;
        var zValue = 0f;

        // Only for polar WASD controls
        // Ratio allows the agent to turn more or less in place, but still turn normally while moving.
        // The higher the ratio, the smaller circle the agent makes while turning in place (A/D)
        const float ratio = 1f;
        
        if (Input.GetKey(KeyCode.W)) zValue = 1f;
        if (Input.GetKey(KeyCode.S)) zValue = -1f;
        
        if (Input.GetKey(KeyCode.D)) xValue = 1f/ratio;
        if (Input.GetKey(KeyCode.A)) xValue = -1f/ratio;

        // if ((Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.S))
        //     &&
        //     (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.D)))
        // {
        //     xValue *= ratio;
        // }
        //
        var force = new Vector3(xValue, 0, zValue);
        // force = transform.rotation * force;
        //
        // Debug.Log(force);
        
        actionsOut[0] = force.x;
        actionsOut[1] = force.z;
    }

    
    public void Freeze()
    {
        Unfrozen = 0;
        Rigidbody.constraints = Rigidbody.constraints |
                                 RigidbodyConstraints.FreezePositionX | 
                                 RigidbodyConstraints.FreezePositionZ | 
                                 RigidbodyConstraints.FreezeRotationY;
        
        Debug.Log("Freezing agent");

        enabled = false;
        // GetComponent<DecisionRequester>().enabled = false;
        // GetComponent<DecisionRequester>().DecisionPeriod = Int32.MaxValue;

    }

    public void Unfreeze()
    {
        enabled = true;
        // GetComponent<DecisionRequester>().enabled = true;
        // GetComponent<DecisionRequester>().DecisionPeriod = 5;

        Debug.Log("Unfreezing agent");

        
        Unfrozen = 1;
        Rigidbody.constraints &= ~RigidbodyConstraints.FreezePositionX;
        Rigidbody.constraints &= ~RigidbodyConstraints.FreezePositionZ;
        Rigidbody.constraints &= ~RigidbodyConstraints.FreezeRotationY;
    }


}
