using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public class Empty : IRewarder
    {
        public float ComputeReward(Transform transform)
        {
            return 0f;
        }

        public float CollisionReward(Transform transform, Collision other, bool stay)
        {
            return 0f;
        }

        public float TriggerReward(Transform transform, Collider other, bool stay)
        {
            return 0f;
        }

        public float ActionReward(Transform transform, ActionBuffers actions)
        {
            return 0f;
        }
    }
}