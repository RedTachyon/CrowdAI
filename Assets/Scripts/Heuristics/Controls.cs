using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Heuristics
{
    public class Controls : IHeuristic
    {
        public void DoAction(in ActionBuffers actionsOut, Transform transform)
        {
            var cActionsOut = actionsOut.ContinuousActions;

            var xValue = 0f;
            var zValue = 0f;
            Vector3 force;

            // Only for polar WASD controls
            // Ratio allows the agent to turn more or less in place, but still turn normally while moving.
            // The higher the ratio, the smaller circle the agent makes while turning in place (A/D)
            const float ratio = 1f;
        
            if (Input.GetKey(KeyCode.W)) xValue = 1f;
            if (Input.GetKey(KeyCode.S)) xValue = -1f;
        
            if (Input.GetKey(KeyCode.D)) zValue = -1f/ratio;
            if (Input.GetKey(KeyCode.A)) zValue = 1f/ratio;

            if (true)
            {
                force = new Vector3(xValue, 0, zValue);
            }

            cActionsOut[0] = force.x;
            cActionsOut[1] = force.z;
        }
    }
}