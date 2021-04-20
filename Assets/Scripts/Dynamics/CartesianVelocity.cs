using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Dynamics
{
    public class CartesianVelocity : IDynamics
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

            var xSpeed = Mathf.Clamp(vectorAction[0], -1f, 1f);
            var zSpeed = Mathf.Clamp(vectorAction[1], -1f, 1f);        

            var velocity = new Vector3(xSpeed, 0, zSpeed);
            
            rigidbody.velocity = Vector3.ClampMagnitude(velocity, maxSpeed);
        }
    }
}