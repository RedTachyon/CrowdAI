using System.Collections.Generic;
using Agents;
using UnityEngine;

namespace Initializers
{
    public class Hallway : IInitializer
    {
        public void PlaceAgents(Transform baseTransform, float size, List<Vector3> obstacles)
        {
            var placedAgents = new List<Vector3>();
            var placedGoals = new List<Vector3>();
            var agentIdx = 0;

            var numAgents = baseTransform.childCount;

            foreach (Transform agent in baseTransform)
            {
                var goal = agent.GetComponent<AgentBasic>().goal;

                Vector3 newPosition;
                Vector3 goalPosition;
                Quaternion newRotation;

                if (agentIdx < numAgents / 2)
                {
                    newPosition = MLUtils.NoncollidingPosition(
                        6f,
                        9f,
                        -4f,
                        4f,
                        agent.localPosition.y,
                        placedAgents
                    );

                    goalPosition = newPosition;
                    goalPosition.x = -goalPosition.x;

                }
                else
                {
                    newPosition = MLUtils.NoncollidingPosition(
                        -4f,
                        4f,
                        6f,
                        9f,
                        agent.localPosition.y,
                        placedAgents
                    );

                    goalPosition = newPosition;
                    goalPosition.z = -goalPosition.z;
                }


                // goalPosition = Quaternion.AngleAxis(180, Vector3.up) * newPosition;
                // goalPosition.y = goal.localPosition.y;
                newRotation = Quaternion.LookRotation(goalPosition, Vector3.up);

                agent.localPosition = newPosition;
                agent.localRotation = newRotation;
                goal.localPosition = goalPosition;

                // Save the placed agents
                placedAgents.Add(newPosition);
                placedGoals.Add(goalPosition);

                // Reset the dynamics
                agent.GetComponent<Rigidbody>().velocity = Vector3.zero;
                agent.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;

                agent.GetComponent<AgentBasic>().PreviousPosition = agent.localPosition;

                agentIdx++;
            }

        }
    }
}