using Agents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public class EnergyPotential : IRewarder
    {
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
            var (normalEnergy, complexEnergy) = MLUtils.EnergyUsage(velocity, lastVelocity, agent.e_s, agent.e_w, dt);
            
            var energy = Params.UseComplexEnergy ? complexEnergy : normalEnergy;
            
            var reward = -Params.EnergyWeight * energy;
            
            var distanceToGoal = MLUtils.FlatDistance(transform.position, agent.Goal.transform.position);
            var previousDistanceToGoal = MLUtils.FlatDistance(agent.PreviousPositionPhysics, agent.Goal.transform.position);

            var c_p = Params.PotentialEnergyScale * Mathf.Sqrt(agent.e_s * agent.e_w);

            var potentialEnergy = c_p * (previousDistanceToGoal - distanceToGoal);
            
            agent.AddRewardPart(energy, "energy");
            agent.AddRewardPart((previousDistanceToGoal - distanceToGoal), "energyPotential");

            return reward + potentialEnergy;

        }
        
        public float FinishReward(Transform transform, bool success)
        {
            var agent = transform.GetComponent<AgentBasic>();
            var position = transform.localPosition;
            var goalPosition = agent.Goal.transform.localPosition;

            var cost = MLUtils.EnergyHeuristic(position, goalPosition, agent.e_s, agent.e_w);
            agent.AddRewardPart(success ? 0f : -cost, "final");
            return success ? 0f : -Params.FinalEnergyWeight * cost;
        }
    }
}