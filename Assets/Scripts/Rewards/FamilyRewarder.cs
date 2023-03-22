using System.Runtime.InteropServices;
using Agents;
using Managers;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public class FamilyRewarder : IRewarder
    {
        public float ComputeReward(Transform transform)
        {
            var reward = 0f;
            // Debug.Log($"Boring reward at timestep {Manager.Instance.Timestep}");
            
            AgentBasic agent = transform.GetComponent<AgentBasic>();
            Transform goal = agent.Goal;
            if (agent.rewardDisabled)
            {
                var currentSpeed = transform.GetComponent<Rigidbody>().velocity.magnitude;
                var speedNorm = Mathf.Pow(currentSpeed, Params.StandstillExponent);
                reward += Params.StandstillWeight * speedNorm;
                agent.AddRewardPart(-speedNorm, "standstill");
            }
            else
            {
                // Speed similarity
                var idealSpeed = Params.RandomEnergy ? Mathf.Sqrt(agent.e_s / agent.e_w) : Params.ComfortSpeed;
                var currentSpeed = transform.GetComponent<Rigidbody>().velocity.magnitude;
                var speedDiff = Mathf.Pow(Mathf.Abs(currentSpeed - idealSpeed), Params.ComfortSpeedExponent);
                
                if (!agent.CollectedGoal)
                {
                    // Only add the speed reward if the agent hasn't collected the goal yet
                    reward += Params.ComfortSpeedWeight * speedDiff;
                    agent.AddRewardPart(-speedDiff, "speed");
                    // Debug.Log($"speedDiff: {speedDiff}");

                }

                reward += Params.StepReward; // Always add this for simplicity
                agent.AddRewardPart(-1, "time");
            }
            
            // TODO: add non-finishing penalty

            return reward;
        }

        public float CollisionReward(Transform transform, Collision other, bool stay)
        {
            float reward = 0;
            AgentBasic agent = transform.GetComponent<AgentBasic>();

            // Penalty only if it's a collision with an obstacle or another agent
            if ((other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent")) && !agent.CollectedGoal)
            {
                reward += Params.Collision;
                agent.AddRewardPart(-1, "collision");
            }

            return reward;
        }

        public float TriggerReward(Transform transform, Collider other, bool stay)
        {
            float reward = 0;
            
            AgentBasic agent = transform.GetComponent<AgentBasic>();

            // Give the goal reward at most only once per episode
            if (!agent.CollectedGoal)
            {
                reward += Params.Goal;
                agent.AddRewardPart(1, "goal");
            }

            return reward;
            
        }

        public float ActionReward(Transform transform, ActionBuffers actions)
        {
            float reward = 0f;
            // Debug.Log($"Action reward at timestep {Manager.Instance.Timestep}");
            var agent = transform.GetComponent<AgentBasic>();
            if (agent.CollectedGoal)
            {
                return 0f;
            }

            var family = agent.Family;
            var lastOrder = family.LastAction;
            
            var velocity = transform.GetComponent<Rigidbody>().velocity;

            var v = new Vector2(velocity.x, velocity.z);
            var a = new Vector2(lastOrder[0], lastOrder[1]);

            var alignment = Vector2.Dot(v.normalized, a.normalized); // [0, 1]
            
            // Debug.Log($"Current alignment: {alignment}, v.x: {v.x}, v.y: {v.y}, a.x: {a.x}, a.y: {a.y}");
            Debug.Log($"Current alignment: {alignment}");
            agent.AddRewardPart(alignment, "alignment");
            reward += Params.AlignmentWeight * alignment;
            
            return reward;
        }
        
        public float LateReward(Transform transform)
        {
            // Debug.Log($"Computing late reward at timestep {Manager.Instance.Timestep}");

            float reward = 0f;
            return reward;
        }
    }
}