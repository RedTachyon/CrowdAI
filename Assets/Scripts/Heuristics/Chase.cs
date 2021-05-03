using Agents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Heuristics
{
    public class Chase : IHeuristic
    {
        public void DoAction(in ActionBuffers actionsOut, Transform transform)
        {
            var cActionsOut = actionsOut.ContinuousActions;

            var agent = transform.GetComponent<Animal>();
            var isPredator = agent.type == AnimalType.Predator;
            float factor = isPredator ? 1 : -1;
            var otherType = isPredator ? AnimalType.Prey : AnimalType.Predator;

            var nearestAgent = agent.FindNearestAgent(otherType);

            var relativePosition = nearestAgent.localPosition - transform.localPosition;

            var velocity = factor * relativePosition.normalized;

            cActionsOut[0] = velocity.x;
            cActionsOut[1] = velocity.z;
        }
    }
}