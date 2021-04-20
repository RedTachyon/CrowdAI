using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Dynamics
{
    public interface IDynamics
    {
        public void ProcessActions(ActionBuffers actions, Rigidbody rigidbody, float moveSpeed, float rotSpeed, float dragFactor, float maxSpeed);
    }

    public enum DynamicsEnum
    {
        CartesianVelocity,
        CartesianAcceleration,
        PolarVelocity,
        PolarAcceleration
    }
}