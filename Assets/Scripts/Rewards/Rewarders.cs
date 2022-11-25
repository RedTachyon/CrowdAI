using Agents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public interface IRewarder
    {
        public float ComputeReward(Transform transform)
        {
            return 0f;
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
            return 0f;
        }
        public float ActionReward(Transform transform, ActionBuffers actions)
        {
            return 0f;
        }
        
        public float FinishReward(Transform transform, bool success)
        {
            return 0f;
        }
    }

    public enum RewardersEnum
    {
        BaseRewarder,
        Empty,
        Energy,
        EnergyComplex,
    }
    
    public static class Mapper
    {
        public static IRewarder GetRewarder(RewardersEnum rewarderType)
        {
            IRewarder rewarder = rewarderType switch
            {
                RewardersEnum.BaseRewarder => new BaseRewarder(),
                RewardersEnum.Empty => new Empty(),
                RewardersEnum.Energy => new Energy(),
                RewardersEnum.EnergyComplex => new EnergyComplex(),
                // RewardersEnum.AnimalChase => new AnimalChase(),
                _ => null
            };

            return rewarder;
        }
    }
}