using Agents;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public class BaseRewarder : IRewarder
    {
        public float ComputeReward(Transform transform)
        {
            var reward = 0f;
            
            AgentBasic agent = transform.GetComponent<AgentBasic>();
            Transform goal = agent.goal;
            if (agent.CollectedGoal)
            {
                var currentSpeed = transform.GetComponent<Rigidbody>().velocity.magnitude;
                var speedNorm = Mathf.Pow(currentSpeed, Params.ComfortSpeedExponent);
                reward += Params.ComfortSpeedWeight * speedNorm;
            }
            else
            {
                var prevDistance = Vector3.Distance(agent.PreviousPosition, goal.localPosition);
                var currentDistance = Vector3.Distance(transform.localPosition, goal.localPosition);

                // Up to ~0.1
                var diff = prevDistance - currentDistance;


                // Speed similarity
                var idealSpeed = Params.ComfortSpeed;
                var currentSpeed = transform.GetComponent<Rigidbody>().velocity.magnitude;
                var speedDiff = Mathf.Pow(Mathf.Abs(currentSpeed - idealSpeed), Params.ComfortSpeedExponent);

                // Debug.Log($"Current speed rmse: {speedDiff}");


                reward += Params.Potential * diff; // Add reward for getting closer to the goal
                reward += Params.ComfortSpeedWeight * speedDiff;
                reward += Params.StepReward;
            }

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

        public float ActionReward(Transform transform, ActionBuffers actions)
        {
            float reward = 0f;
            return reward;
        }
    }
}