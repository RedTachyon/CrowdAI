using Agents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public class EnergyComplex : IRewarder
    {
        public float FinishReward(Transform transform, bool success)
        {
            var agent = transform.GetComponent<AgentBasic>();
            var position = transform.localPosition;
            var goalPosition = agent.Goal.transform.localPosition;

            var cost = MLUtils.EnergyHeuristic(position, goalPosition, agent.e_s, agent.e_w);
            agent.AddRewardPart(success ? 0f : -cost, "final");
            return success ? 0f : -Params.FinalEnergyWeight * cost;
        }

        public float LateReward(Transform transform)
        {
            var agent = transform.GetComponent<AgentBasic>();
            var dt = Time.fixedDeltaTime;
            if (agent.CollectedGoal)
            {
                return 0f;
            }
            
            // var velocity = agent.Rigidbody.velocity;
            // var lastVelocity = agent.PreviousVelocityPhysics;
            var velocity = (transform.localPosition - agent.PreviousPositionPhysics) / Time.fixedDeltaTime;
            var lastVelocity = (agent.PreviousPositionPhysics - agent.PreviouserPositionPhysics) / Time.fixedDeltaTime;
            var (_, complexEnergy) = MLUtils.EnergyUsage(velocity, lastVelocity, agent.e_s, agent.e_w, dt);
            
            var reward = -Params.EnergyWeight * complexEnergy;
            
            agent.AddRewardPart(complexEnergy, "energy");

            return reward;
            
        }
    }
}