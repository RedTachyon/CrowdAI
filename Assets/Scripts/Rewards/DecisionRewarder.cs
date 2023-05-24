using System;
using System.Runtime.InteropServices;
using Agents;
using Managers;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public class DecisionRewarder : IRewarder
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
            var idealSpeed = Params.RandomEnergy ? Mathf.Sqrt(agent.e_s / agent.e_w) : Params.ComfortSpeed;
            var currentSpeed = velocity.magnitude;
            
            var speedDiff = currentSpeed - idealSpeed;
            var absSpeedDiff = Mathf.Abs(speedDiff);
            var reluSpeedDiff = Mathf.Max(speedDiff, 0);

            var absSpeedPenalty = Mathf.Pow(absSpeedDiff, Params.RewSpeedExp);
            var reluSpeedPenalty = Mathf.Pow(reluSpeedDiff, Params.RewSpeedExp);
            
            
            // Xu et al.
            var goalVector = (goal.localPosition - transform.localPosition);
            goalVector.y = 0;
            goalVector = goalVector.normalized;
            var velocityDiff = velocity - idealSpeed * goalVector;
            var velocityPenalty = velocityDiff.sqrMagnitude;
            var expVelocityPenalty = Mathf.Exp(Params.RewExpVelSigma * velocityDiff.magnitude);

            var diffPotential = Vector3.Dot(velocity, goalVector);

            
            var r_bmr = -agent.e_s * dt;
            var r_drag = -agent.e_w * velocity.sqrMagnitude * dt;
            var r_dynamics = -dynamicsEnergy * dt;
            var r_potential = 2 * Mathf.Sqrt(agent.e_s * agent.e_w) * posDiff;
            var r_diffPotential = 2 * Mathf.Sqrt(agent.e_s * agent.e_w) * diffPotential * dt;
            var r_speedmatch = -absSpeedPenalty * dt;
            var r_speeding = -reluSpeedPenalty * dt;
            var r_velocity = -velocityPenalty * dt * agent.e_w;
            var r_expVelocity = -expVelocityPenalty * dt;
            
            // Debug.Log($"r_dynamics: {r_dynamics}, velocity: {velocity.magnitude}, acceleration: {acceleration.magnitude}, prevVelocity: {prevVelocity.magnitude}, dynamicsEnergy: {dynamicsEnergy}");

            reward += Params.RewBMR * r_bmr;
            agent.AddRewardPart(r_bmr, "r_bmr");

            reward += Params.RewDrag * r_drag;
            agent.AddRewardPart(r_drag, "r_drag");

            reward += Params.RewDyn * r_dynamics;
            agent.AddRewardPart(r_dynamics, "r_dynamics");

            reward += Params.RewPot * r_potential;
            agent.AddRewardPart(r_potential, "r_potential");
            
            reward += Params.RewDiffPot * r_diffPotential;
            agent.AddRewardPart(r_diffPotential, "r_diffPotential");

            reward += Params.RewSpeed * r_speedmatch;
            agent.AddRewardPart(r_speedmatch, "r_speedmatch");

            reward += Params.RewSpeeding * r_speeding;
            agent.AddRewardPart(r_speeding, "r_speeding");

            reward += Params.RewVel * r_velocity;
            agent.AddRewardPart(r_velocity, "r_velocity");

            reward += Params.RewExpVel * r_expVelocity;
            agent.AddRewardPart(r_expVelocity, "r_expVelocity");
            
            

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
                reward += Params.RewGoal;
                agent.AddRewardPart(1, "goal");
            }

            return reward;
            
        }

        public float ActionReward(Transform transform, ActionBuffers actions)
        {
            float reward = 0f;
            return reward;
        }
        
        public float FinishReward(Transform transform, bool success)
        {
            var agent = transform.GetComponent<AgentBasic>();
            float reward = 0f;
            if (success) return reward;

            var penalty = -MLUtils.EnergyHeuristic(transform.localPosition, agent.Goal.localPosition, agent.e_s, agent.e_w);
            
            // var finalDistance = MLUtils.FlatDistance(transform.localPosition, agent.Goal.localPosition);

            // var finalReward = -2 * Mathf.Sqrt(agent.e_s * agent.e_w * finalDistance);
            // var finalReward = -2 * Mathf.Sqrt(agent.e_s * agent.e_w) * finalDistance;

            reward += Params.RewFinal * penalty;
            agent.AddRewardPart(penalty, "final");
            
            // TODO: Instead of assuming the optimal velocity, use the average velocity across the trajectory so far
            // TODO: Track both of them as a metric, but add a switch to choose which one to use for the reward
            
            return reward;
        }
    }
}