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
            var dt = Manager.Instance.DecisionDeltaTime;
            
            AgentBasic agent = transform.GetComponent<AgentBasic>();
            Transform goal = agent.Goal;
            
            if (agent.rewardDisabled) return reward;
            if (agent.CollectedGoal) return reward;
            
            var prevDistance = MLUtils.FlatDistance(agent.PreviousPosition, goal.localPosition);
            var currentDistance = MLUtils.FlatDistance(transform.localPosition, goal.localPosition);
            var posDiff = (prevDistance - currentDistance);
            

            var velocity = agent.Rigidbody.velocity;
            var prevVelocity = agent.PreviousVelocity;
            var acceleration = (velocity - agent.PreviousVelocity) / dt;
            

            // Debug.Log($"Acceleration: {acceleration.magnitude}, prevVelocity: {prevVelocity.magnitude}, velocity: {velocity.magnitude}");

            var dynamicsEnergy = Mathf.Abs(
                    Vector3.Dot(velocity, acceleration)
                    + agent.e_w * Vector3.Dot(velocity, prevVelocity)
                );
            

            // Speed similarity
            
            
            var r_bmr = -agent.e_s * dt;
            var r_dynamics = -dynamicsEnergy * dt;
            
            // Debug.Log($"r_dynamics: {r_dynamics}, velocity: {velocity.magnitude}, acceleration: {acceleration.magnitude}, prevVelocity: {prevVelocity.magnitude}, dynamicsEnergy: {dynamicsEnergy}");

            reward += Params.RewBMR * r_bmr;
            agent.AddRewardPart(r_bmr, "r_bmr");


            reward += Params.RewDyn * r_dynamics;
            agent.AddRewardPart(r_dynamics, "r_dynamics");

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

            var alignment = 2*Mathf.Sqrt(agent.e_s * agent.e_w) * Vector2.Dot(v, a.normalized) * Time.fixedDeltaTime;
            Debug.Log($"Alignment: {alignment}");
            
            // Debug.Log($"Current alignment: {alignment}, v.x: {v.x}, v.y: {v.y}, a.x: {a.x}, a.y: {a.y}");
            // Debug.Log($"Current alignment: {alignment}");
            
            agent.AddRewardPart(alignment, "r_alignment");
            reward += Params.AlignmentWeight * alignment;
            
            
            
            return reward;
        }
        
        public float LateReward(Transform transform)
        {
            return 0f;

        }
    }
}