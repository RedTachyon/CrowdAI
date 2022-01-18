using System;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Dynamics
{
    public class PolarVelocity : IDynamics
    {
        public void ProcessActions(
            ActionBuffers actions,
            Rigidbody rigidbody,
            float maxSpeed, 
            float maxAccel, // unused
            float rotSpeed,
            Func<Vector2, Vector2> squasher)
        {

            var vectorAction = new Vector2(actions.ContinuousActions[0], actions.ContinuousActions[1]);
            vectorAction = Squasher.Tanh(vectorAction);
            

            var linearSpeed = Mathf.Max(vectorAction.y, 0f);
            var angularSpeed = vectorAction.x;

            rigidbody.MoveRotation(Quaternion.Euler(0, angularSpeed * rotSpeed, 0) * rigidbody.rotation);
            
            
            var newVelocity = rigidbody.rotation * Vector3.forward * linearSpeed * maxSpeed;
            rigidbody.velocity = newVelocity;
        
        }
    }
}