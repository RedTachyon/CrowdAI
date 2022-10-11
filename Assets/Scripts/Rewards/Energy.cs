using Agents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public class Energy : IRewarder
    {
        public float ComputeReward(Transform transform)
        {
            throw new System.NotImplementedException();
        }

        public float CollisionReward(Transform transform, Collision other, bool stay)
        {
            throw new System.NotImplementedException();
        }

        public float TriggerReward(Transform transform, Collider other, bool stay)
        {
            throw new System.NotImplementedException();
        }

        public float ActionReward(Transform transform, ActionBuffers actions)
        {
            var energySpent = 0f;
            var agent = transform.GetComponent<AgentBasic>();
            var velocity = agent.Rigidbody.velocity;
            var lastVelocity = agent.PreviousVelocity;
            var speedSqr = velocity.sqrMagnitude;
            var speed = Mathf.Sqrt(speedSqr);
            var acceleration = (velocity - lastVelocity).magnitude / Time.fixedDeltaTime;
            energySpent += agent.e_s * Time.fixedDeltaTime;
            energySpent += agent.e_w * speedSqr * Time.fixedDeltaTime;
            energySpent += (Params.EnergyComplex ? 1f : 0f) * acceleration * speed * Time.fixedDeltaTime;
            return energySpent;
        }
    }
}