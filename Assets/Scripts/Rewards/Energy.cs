using Agents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public class Energy : IRewarder
    {
        public float FinishReward(Transform transform, bool success)
        {
            var agent = transform.GetComponent<AgentBasic>();
            var position = transform.localPosition;
            var goalPosition = agent.Goal.transform.localPosition;

            var cost = MLUtils.EnergyHeuristic(position, goalPosition, agent.e_s, agent.e_w);
            return success ? 0f : -Params.FinalEnergyWeight * cost;
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
            var (energy, complexEnergy) = MLUtils.EnergyUsage(velocity, lastVelocity, agent.e_s, agent.e_w, dt);

            return -Params.EnergyWeight * energy;
            // var agent = transform.GetComponent<AgentBasic>();
            //
            // if (agent.CollectedGoal)
            // {
            //     return 0f;
            // }
            // var energySpent = 0f;
            // var velocity = agent.Rigidbody.velocity;
            // var lastVelocity = agent.PreviousVelocityPhysics;
            // var speedSqr = velocity.sqrMagnitude;
            // energySpent += agent.e_s * Time.fixedDeltaTime;
            // energySpent += agent.e_w * speedSqr * Time.fixedDeltaTime;
            // return energySpent;
        }
    }
}