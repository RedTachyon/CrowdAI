using System;
using System.Linq;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Dynamics
{
    public class CartesianVelocity : IDynamics
    {
        public void ProcessActions(
            ActionBuffers actions,
            Rigidbody rigidbody,
            float maxSpeed, 
            float maxAccel, // unused
            float rotSpeed, // unused
            Func<Vector2, Vector2> squasher
        )
        {

            var vectorAction = new Vector2(actions.ContinuousActions[0], actions.ContinuousActions[1]);
            vectorAction = Squasher.RadialTanh(vectorAction);
            
            
            var velocity = new Vector3(vectorAction.x, 0, vectorAction.y);
            velocity *= maxSpeed;
            // velocity = velocity.normalized * moveSpeed;
            // velocity = Vector3.ClampMagnitude(velocity, maxSpeed);

            rigidbody.velocity = velocity;
            
            if (velocity.magnitude > 1e-6)
            {
                rigidbody.rotation = Quaternion.LookRotation(velocity, Vector3.up);
            }

        }
        
    }
}