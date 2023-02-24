using System.Runtime.InteropServices;
using Agents;
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
                var prevDistance = MLUtils.FlatDistance(agent.PreviousPosition, goal.localPosition);
                var currentDistance = MLUtils.FlatDistance(transform.localPosition, goal.localPosition);

                // Up to ~0.1
                var diff = prevDistance - currentDistance;

                // Speed similarity
                var idealSpeed = Params.RandomEnergy ? Mathf.Sqrt(agent.e_s / agent.e_w) : Params.ComfortSpeed;
                var currentSpeed = transform.GetComponent<Rigidbody>().velocity.magnitude;
                var speedDiff = Mathf.Pow(Mathf.Abs(currentSpeed - idealSpeed), Params.ComfortSpeedExponent);
                
                

                reward += Params.Potential * diff; // Add this on the last step too
                agent.AddRewardPart(diff, "potential");
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
            return reward;
        }
    }
}