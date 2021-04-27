using UnityEngine;

namespace Rewards
{
    public class PreyChase : IRewarder
    {
        public float ComputeReward(Transform transform)
        {
            throw new System.NotImplementedException();
        }

        public float CollisionReward(Transform transform, Collision other, bool stay)
        {
            return 0f;
        }

        public float TriggerReward(Transform transform, Collider other, bool stay)
        {
            return 0f;
        }
    }
}