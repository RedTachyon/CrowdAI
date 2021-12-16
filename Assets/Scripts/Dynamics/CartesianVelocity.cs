using System;
using System.Linq;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Dynamics
{
    public class CartesianVelocity : IDynamics
    {
        public void ProcessActions(
            // unused
            ActionBuffers actions,
            Rigidbody rigidbody,
            float maxSpeed, 
            float maxAccel, // unused
            float rotSpeed, // unused
            Func<Vector2, Vector2> squasher
            // unused
        )
        {

            var vectorAction = new Vector2(actions.ContinuousActions[0], actions.ContinuousActions[1]);
            vectorAction = squasher(vectorAction);
            
            
            var velocity = new Vector3(vectorAction.x, 0, vectorAction.y);
            velocity = velocity * maxSpeed;
            // velocity = velocity.normalized * moveSpeed;
            // velocity = Vector3.ClampMagnitude(velocity, maxSpeed);

            rigidbody.velocity = velocity;

        }
        
    }
}