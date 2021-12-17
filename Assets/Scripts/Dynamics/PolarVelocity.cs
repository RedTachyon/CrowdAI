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

            var vectorAction = new Vector2(actions.ContinuousActions[0], actions.ContinuousActions[1]);
            vectorAction = squasher(vectorAction);
            

            var linearSpeed = vectorAction.x;
            
            var transform = rigidbody.transform;
        
            // Debug.Log($"Taking action: {vectorAction[0]}, {vectorAction[1]}");

            var angularSpeed = Mathf.Clamp(vectorAction[0], -1f, 1f);
            // var linearSpeed = Mathf.Clamp(vectorAction[1], -1f, 1f);
            
            // Apply the rotation
            // var timeFactor = Time.fixedDeltaTime / 0.02f; // Simulation is balanced around 0.02
            Vector3 rotation = transform.rotation.eulerAngles + Vector3.up * angularSpeed * rotSpeed;
            rigidbody.rotation = Quaternion.Euler(rotation);
            
            
            var newVelocity = rigidbody.transform.forward * linearSpeed * maxAccel; // Rough adjustment to a normal range
            rigidbody.velocity = newVelocity;
        
        }
    }
}