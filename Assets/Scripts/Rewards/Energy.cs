using Agents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public class Energy : IRewarder
    {
        public float ComputeReward(Transform transform)
        {
            return 0f;
        }

        public float CollisionReward(Transform transform, Collision other, bool stay)
        {
            return 0f;
        }

        public float TriggerReward(Transform transform, Collider other, bool stay)
        {
            return 0f;
        }

        public float ActionReward(Transform transform, ActionBuffers actions)
        {
            var agent = transform.GetComponent<AgentBasic>();

            if (agent.CollectedGoal)
            {
                return 0f;
            }
            var energySpent = 0f;
            var velocity = agent.Rigidbody.velocity;
            var lastVelocity = agent.PreviousVelocityPhysics;
            var speedSqr = velocity.sqrMagnitude;
            energySpent += agent.e_s * Time.fixedDeltaTime;
            energySpent += agent.e_w * speedSqr * Time.fixedDeltaTime;
            return energySpent;
        }
    }
}