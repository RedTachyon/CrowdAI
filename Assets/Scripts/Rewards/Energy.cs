using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Rewards
{
    public class Energy : IRewarder
    {
        public float ComputeReward(Transform transform)
        {
            throw new System.NotImplementedException();
        }

        public float CollisionReward(Transform transform, Collision other, bool stay)
        {
            throw new System.NotImplementedException();
        }

        public float TriggerReward(Transform transform, Collider other, bool stay)
        {
            throw new System.NotImplementedException();
        }

        public float ActionReward(Transform transform, ActionBuffers actions)
        {
            throw new System.NotImplementedException();
        }
    }
}