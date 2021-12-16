using System;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Dynamics
{
    public interface IDynamics
    {
        public void ProcessActions(ActionBuffers actions, Rigidbody rigidbody, float maxSpeed, float maxAccel,
            float rotSpeed, Func<Vector2, Vector2> squasher);
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