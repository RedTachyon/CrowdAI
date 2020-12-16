using System;
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


    public override void Initialize()
    {
        Rigidbody = GetComponent<Rigidbody>();
    }
    

    public override void OnActionReceived(float[] vectorAction)
    {
        
        // Forward velocity
        var linearSpeed = Unfrozen * Mathf.Clamp(vectorAction[0], -1f, 1.0f);
        
        // Angular velocity
        var angularSpeed = Unfrozen * Mathf.Clamp(vectorAction[1], -1f, 1f);
        
        // Apply the force
        Vector3 force = transform.forward * linearSpeed * moveSpeed;
        
        // Reduce the velocity friction-like
        Vector3 drag = -dragFactor * Rigidbody.velocity;
        Rigidbody.AddForce(force + drag);

        // Rigidbody.velocity = force / 10f;

        // Apply the rotation
        Vector3 rotation = transform.rotation.eulerAngles + Vector3.up * angularSpeed * rotationSpeed;
        // _rigidbody.MoveRotation(Quaternion.Euler(rotation));

        Rigidbody.rotation = Quaternion.Euler(rotation);
        
        // Rigidbody.velocity = new Vector3(linearSpeed, 0f, angularSpeed) * moveSpeed;
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
