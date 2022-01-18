using System;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Dynamics
{
    public class CartesianAcceleration : IDynamics
    {
        public void ProcessActions(// a_m maximum acceleration
            ActionBuffers actions,
            Rigidbody rigidbody,
            float maxSpeed, 
            float maxAccel,
            float rotSpeed, // unused
            Func<Vector2, Vector2> squasher)
        {

            var vectorAction = new Vector2(actions.ContinuousActions[0], actions.ContinuousActions[1]);
            vectorAction = Squasher.RadialTanh(vectorAction);  // ||X|| <= 1

            // var xForce = Mathf.Clamp(vectorAction[0], -1f, 1f);
            // var zForce = Mathf.Clamp(vectorAction[1], -1f, 1f);        
            

            var force = new Vector3(vectorAction.x, 0, vectorAction.y) * maxAccel;
            var dragFactor = maxAccel / maxSpeed;
            
            var currentVelocity = rigidbody.velocity;
            var newVelocity = currentVelocity + (force - dragFactor * currentVelocity) * Time.fixedDeltaTime;
            
            rigidbody.velocity = newVelocity.magnitude > 1e-3f ? newVelocity : Vector3.zero; 
            
            if (rigidbody.velocity.magnitude > 1e-6)
            {
                rigidbody.rotation = Quaternion.LookRotation(rigidbody.velocity, Vector3.up);
            }
            
            // rigidbody.AddForce(force + drag);
            
            // rigidbody.velocity = Vector3.ClampMagnitude(velocity, maxSpeed);
        }
        
    }
}