using Agents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public interface IRewarder
    {
        public float ComputeReward(Transform transform);
        public float CollisionReward(Transform transform, Collision other, bool stay);
        public float TriggerReward(Transform transform, Collider other, bool stay);
        public float ActionReward(Transform transform, ActionBuffers actions);
    }

    public enum RewardersEnum
    {
        BaseRewarder,
        Empty,
        AnimalChase,
    }
    
    public static class Mapper
    {
        public static IRewarder GetRewarder(RewardersEnum rewarderType)
        {
            IRewarder rewarder = rewarderType switch
            {
                RewardersEnum.BaseRewarder => new BaseRewarder(),
                RewardersEnum.Empty => new Empty(),
                // RewardersEnum.AnimalChase => new AnimalChase(),
                _ => null
            };

            return rewarder;
        }
    }
}