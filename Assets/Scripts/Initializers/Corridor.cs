using System.Collections.Generic;
using Agents;
using UnityEngine;

namespace Initializers
{
    public class Corridor : IInitializer
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

                        newPosition = MLUtils.NoncollidingPosition(
                            -6f,
                            -9.4f,
                            -3f,
                            3f,
                            agent.localPosition.y,
                            placedAgents
                        );

                        goalPosition = newPosition;
                        goalPosition.x = -goalPosition.x;
                        

                        // goalPosition = Quaternion.AngleAxis(180, Vector3.up) * newPosition;
                        // goalPosition.y = goal.localPosition.y;
                        newRotation = Quaternion.LookRotation(Vector3.right, Vector3.up);

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