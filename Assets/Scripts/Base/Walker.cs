using System;
using System.Collections.Specialized;
using Dynamics;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.Serialization;


public class Walker : Agent
{
    // A basic agent that only implements the environment dynamics, i.e. walking around with friction
    protected Rigidbody Rigidbody;

    // public bool velocityControl = false;
    public float moveSpeed = 25f;
    public float rotationSpeed = 3f;
    
    public DynamicsEnum DynamicsType;
    internal IDynamics Dynamics;

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

        Dynamics = DynamicsType switch
        {
            DynamicsEnum.CartesianVelocity => new CartesianVelocity(),
            DynamicsEnum.CartesianAcceleration => new CartesianAcceleration(),
            DynamicsEnum.PolarVelocity => new PolarVelocity(),
            DynamicsEnum.PolarAcceleration => new PolarAcceleration(),
            _ => Dynamics
        };
    }


    public override void OnActionReceived(ActionBuffers actions)
    {
        base.OnActionReceived(actions);
        // Debug.Log($"{name} OnAction at step {GetComponentInParent<Statistician>().Time}");
        Dynamics.ProcessActions(actions, Rigidbody, moveSpeed, rotationSpeed, dragFactor, 3f);
        

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
        
        if (Input.GetKey(KeyCode.W)) xValue = 1f;
        if (Input.GetKey(KeyCode.S)) xValue = -1f;
        
        if (Input.GetKey(KeyCode.D)) zValue = 1f/ratio;
        if (Input.GetKey(KeyCode.A)) zValue = -1f/ratio;

        if (true)
        {
            force = new Vector3(xValue, 0, zValue);
        }
        // else
        // {   
        //     force = goal.position - transform.position;
        //     force.y = 0f;
        //     force = force.normalized;
        // }

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
