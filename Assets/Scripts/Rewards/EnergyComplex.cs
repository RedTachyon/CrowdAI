using Agents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public class EnergyComplex : IRewarder
    {
        public float ComputeReward(Transform transform)
        {
            return 0;
        }

        public float CollisionReward(Transform transform, Collision other, bool stay)
        {
            return 0;
        }

        public float TriggerReward(Transform transform, Collider other, bool stay)
        {
            return 0;
        }

        public float ActionReward(Transform transform, ActionBuffers actions)
        {
            var agent = transform.GetComponent<AgentBasic>();
            var dt = Time.fixedDeltaTime;
            if (agent.CollectedGoal)
            {
                return 0f;
            }
            var velocity = agent.Rigidbody.velocity;
            var lastVelocity = agent.PreviousVelocity;
            var speed = velocity.magnitude;
            var lastSpeed = lastVelocity.magnitude;
            var acceleration = (velocity - lastVelocity).magnitude / dt;
            
            var speed_threshold = (1f - agent.e_w * dt) * lastSpeed;
            var a_p = (speed - speed_threshold) / dt;

            var energySpent = 0f;
            energySpent += agent.e_s * Time.fixedDeltaTime;
            
            if (speed >= lastSpeed) // TODO: Triple check this math
            {
                energySpent += agent.e_w * speed * speed * dt;
                energySpent += acceleration * speed;
            } else if (speed >= speed_threshold)
            {
                energySpent += a_p * speed;
            } else  // speed < speed_threshold
            {
                energySpent -= a_p * speed_threshold;
            }
            
            // TODO: compute and implement an end-of-episode energy penalty

            energySpent += agent.e_w * speed * speed * Time.fixedDeltaTime;
            energySpent += acceleration * speed * Time.fixedDeltaTime;
            return energySpent;
        }
    }
}