using System;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Dynamics
{
    public class PolarVelocity : IDynamics
    {
        public void ProcessActions(ActionBuffers actions,
            Rigidbody rigidbody,
            float maxSpeed, float maxAccel,
            float rotSpeed,
            Func<Vector2, Vector2> squasher)
        {

            var vectorAction = actions.ContinuousActions;
            var transform = rigidbody.transform;
        
            // Debug.Log($"Taking action: {vectorAction[0]}, {vectorAction[1]}");

            var angularSpeed = Mathf.Clamp(vectorAction[0], -1f, 1f);
            var linearSpeed = Mathf.Clamp(vectorAction[1], -1f, 1f);

            // var xSpeed = Unfrozen * Mathf.Clamp(vectorAction[0], -1f, 1f);
            // var zSpeed = Unfrozen * Mathf.Clamp(vectorAction[1], -1f, 1f);        
        
            // var velocity = rigidbody.velocity;
        
            // Debug.Log(velocity);

            // Average mean comfort speed = 1.4mps
        
            // Apply the rotation
            // var timeFactor = Time.fixedDeltaTime / 0.02f; // Simulation is balanced around 0.02
            Vector3 rotation = transform.rotation.eulerAngles + Vector3.up * angularSpeed * rotSpeed;
            rigidbody.rotation = Quaternion.Euler(rotation);
            
            
            var newVelocity = rigidbody.transform.forward * linearSpeed * maxAccel; // Rough adjustment to a normal range
            rigidbody.velocity = newVelocity;
        
        }
    }
}