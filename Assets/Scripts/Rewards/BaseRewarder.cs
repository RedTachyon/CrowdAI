using Agents;
using Unity.MLAgents;
using UnityEngine;

namespace Rewards
{
    public class BaseRewarder : IRewarder
    {
        public float ComputeReward(Transform transform)
        {
            AgentBasic agent = transform.GetComponent<AgentBasic>();
            Transform goal = agent.goal;
            
            var prevDistance = Vector3.Distance(agent.PreviousPosition, goal.localPosition);
            var currentDistance = Vector3.Distance(transform.localPosition, goal.localPosition);
            
            // Up to ~0.1
            var diff = prevDistance - currentDistance;
            
            return Params.Potential * diff;  // Add reward for getting closer to the goal

        }

        public float CollisionReward(Transform transform, Collision other, bool stay)
        {
            float reward = 0;
            
            // Penalty only if it's a collision with an obstacle or another agent
            if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
            {
                reward += Params.Collision;
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
            }

            return reward;
            
        }
    }
}