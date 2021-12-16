using System;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Dynamics
{
    public class PolarAcceleration : IDynamics
    {
        public void ProcessActions(ActionBuffers actions,
            Rigidbody rigidbody,
            float maxSpeed, float maxAccel,
            float rotSpeed,
            Func<Vector2, Vector2> squasher)
        {

            var vectorAction = actions.ContinuousActions;
            var transform = rigidbody.transform;

            var dragFactor = maxAccel / maxSpeed;
        
            // Debug.Log($"Taking action: {vectorAction[0]}, {vectorAction[1]}");

            var linearForce = Mathf.Clamp(vectorAction[0], -0.5f, 1f);
            var angularSpeed = Mathf.Clamp(vectorAction[1], -1f, 1f);

            // var xSpeed = Unfrozen * Mathf.Clamp(vectorAction[0], -1f, 1f);
            // var zSpeed = Unfrozen * Mathf.Clamp(vectorAction[1], -1f, 1f);        
        
            var velocity = rigidbody.velocity;
        
            // Debug.Log(velocity);

            // Apply the force

            var force = transform.forward * linearForce * maxAccel;
            var drag = -dragFactor * velocity;
            rigidbody.AddForce(force + drag);
            
            // Apply the rotation
            var timeFactor = Time.fixedDeltaTime / 0.02f; // Simulation is balanced around 0.02
            
            Vector3 rotation = transform.rotation.eulerAngles + Vector3.up * angularSpeed * rotSpeed * timeFactor;
            rigidbody.rotation = Quaternion.Euler(rotation);

            rigidbody.velocity = Vector3.ClampMagnitude(rigidbody.velocity, maxSpeed);
        }
    }
}