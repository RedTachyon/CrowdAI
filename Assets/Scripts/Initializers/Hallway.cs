using System.Collections.Generic;
using Agents;
using Managers;
using UnityEngine;

namespace Initializers
{
    public class Hallway : IInitializer
    {
        private readonly Transform _ownObstacles;
        
        public Hallway()
        {
            _ownObstacles = Manager.Instance.AllObstacles.Find("Hallway");
        }
        public void PlaceAgents(Transform baseTransform, float size, List<Vector3> obstacles)
        {
            _ownObstacles.gameObject.SetActive(true);
            
            var placedAgents = new List<Vector3>();
            var placedGoals = new List<Vector3>();
            var agentIdx = 0;

            var numAgents = baseTransform.childCount;

            foreach (Transform agent in baseTransform)
            {
                var goal = agent.GetComponent<AgentBasic>().goal;

                Vector3 newPosition;
                Vector3 goalPosition;
                Vector3 goalOffset;
                Quaternion newRotation;
                
                float xMin, xMax, zMin, zMax;

                if (agentIdx < numAgents / 2)
                {
                    xMin = -9f;
                    xMax = -6f;
                    zMin = -1.5f;
                    zMax = 1.5f;
                    goalOffset = new Vector3(15f, 0, 0);
                    newRotation = Quaternion.LookRotation(Vector3.right);
                }
                else
                {
                    xMin = -1.5f;
                    xMax = 1.5f;
                    zMin = -9f;
                    zMax = -6f;
                    goalOffset = new Vector3(0, 0, 15f);
                    newRotation = Quaternion.LookRotation(Vector3.forward);
                }

                if (Params.GridSpawn)
                {
                    var gridSize = numAgents / 2;
                    if (numAgents % 2 == 1)
                    {
                        gridSize++;
                    }
                    var idx = agentIdx % gridSize;
                    newPosition = 
                        MLUtils.GridPosition(xMin, xMax, zMin, zMax, agent.localPosition.y, idx, gridSize);

                }
                else
                {
                    newPosition =
                        MLUtils.NoncollidingPosition(xMin, xMax, zMin, zMax, agent.localPosition.y, placedAgents);
                }

                goalPosition = newPosition + goalOffset;
                
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