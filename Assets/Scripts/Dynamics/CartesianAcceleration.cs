using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Dynamics
{
    public class CartesianAcceleration : IDynamics
    {
        public void ProcessActions(
            ActionBuffers actions,
            Rigidbody rigidbody,
            float moveSpeed,
            float rotSpeed,
            float dragFactor = 0f,
            float maxSpeed = float.PositiveInfinity)
        {

            var vectorAction = actions.ContinuousActions;

            var xForce = Mathf.Clamp(vectorAction[0], -1f, 1f);
            var zForce = Mathf.Clamp(vectorAction[1], -1f, 1f);        
        
            var velocity = rigidbody.velocity;

            var force = new Vector3(xForce, 0, zForce) * moveSpeed;
            var drag = -dragFactor * velocity;
            rigidbody.AddForce(force + drag);
            
            rigidbody.velocity = Vector3.ClampMagnitude(velocity, maxSpeed);
        }
    }
}