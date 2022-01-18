using System;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Dynamics
{
    public class PolarAcceleration : IDynamics
    {
        public void ProcessActions(
            ActionBuffers actions,
            Rigidbody rigidbody,
            float maxSpeed, 
            float maxAccel,
            float rotSpeed,
            Func<Vector2, Vector2> squasher)
        {

            var vectorAction = new Vector2(actions.ContinuousActions[0], actions.ContinuousActions[1]);
            vectorAction = Squasher.Tanh(vectorAction);
            

            var linearSpeed = vectorAction.y;
            var angularSpeed = vectorAction.x;

            rigidbody.MoveRotation(Quaternion.Euler(0, angularSpeed * rotSpeed, 0) * rigidbody.rotation);
            
            var force = linearSpeed * maxAccel;
            var dragFactor = maxAccel / maxSpeed;
            
            var currentSpeed = rigidbody.velocity.magnitude;
            var newSpeed = currentSpeed + (force - dragFactor * currentSpeed) * Time.fixedDeltaTime;
            newSpeed = Mathf.Max(newSpeed, 0f);
            
            
            var newVelocity = rigidbody.transform.forward * newSpeed;
            rigidbody.velocity = newVelocity;
            
        }
    }
}