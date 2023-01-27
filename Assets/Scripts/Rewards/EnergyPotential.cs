using Agents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public class EnergyPotential : IRewarder
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
            
            var distanceToGoal = Vector3.Distance(transform.position, agent.Goal.transform.position);
            var previousDistanceToGoal = Vector3.Distance(agent.PreviousPositionPhysics, agent.Goal.transform.position);
            
            var c_p = 4 * Mathf.Sqrt(agent.e_s * agent.e_w);
            
            var potentialEnergy = c_p * (previousDistanceToGoal - distanceToGoal);
            
            agent.AddRewardPart(complexEnergy, "energy");
            agent.AddRewardPart(potentialEnergy, "energyPotential");

            return reward;
            
        }
    }
}