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
            
            var speedSign = Vector3.Dot(rigidbody.velocity, rigidbody.transform.forward) > 0 ? 1f : -1f;
            speedSign = Params.BackwardsAllowed ? speedSign : 1f;
            
            var currentSpeed = speedSign * rigidbody.velocity.magnitude;
            // var dragSign = Vector3.Dot(rigidbody.velocity, rigidbody.transform.forward) > 0 ? 1f : -1f;
            // var dragSign = 0f;
            var deltaSpeed = force - dragFactor * currentSpeed;
            var newSpeed = currentSpeed + deltaSpeed * Time.fixedDeltaTime;
            // newSpeed = Mathf.Max(newSpeed, 0f);
            
            // Note - either include fixedDeltaTime and then Impulse or Velocity change, or not and then Acceleration/Force
            // rigidbody.AddForce(rigidbody.transform.forward * deltaSpeed, ForceMode.Force);
            
            
            var newVelocity = rigidbody.transform.forward * newSpeed;
            // Debug.Log(newVelocity);
            rigidbody.velocity = newVelocity;
            
        }
    }
}