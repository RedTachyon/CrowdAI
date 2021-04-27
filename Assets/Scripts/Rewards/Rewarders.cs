using UnityEngine;

namespace Rewards
{
    public interface IRewarder
    {
        public float ComputeReward(Transform transform);
        public float CollisionReward(Transform transform, Collision other, bool stay);
        public float TriggerReward(Transform transform, Collider other, bool stay);
    }

    public enum RewardersEnum
    {
        BaseRewarder
    }
}