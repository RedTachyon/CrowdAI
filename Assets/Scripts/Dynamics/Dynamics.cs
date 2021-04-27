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
    
    public static class Mapper
    {
        public static IDynamics GetDynamics(DynamicsEnum dynamicsType)
        {
            IDynamics dynamics = dynamicsType switch
            {
                DynamicsEnum.CartesianVelocity => new CartesianVelocity(),
                DynamicsEnum.CartesianAcceleration => new CartesianAcceleration(),
                DynamicsEnum.PolarVelocity => new PolarVelocity(),
                DynamicsEnum.PolarAcceleration => new PolarAcceleration(),
                _ => null
            };

            return dynamics;
        }
    }
}