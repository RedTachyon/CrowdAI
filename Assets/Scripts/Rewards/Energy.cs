using Agents;
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
            var energySpent = 0f;
            var agent = transform.GetComponent<AgentBasic>();
            energySpent += Params.E_s * Time.fixedDeltaTime;
            energySpent += Params.E_w * agent.Rigidbody.velocity.sqrMagnitude * Time.fixedDeltaTime;
            return energySpent;
        }
    }
}