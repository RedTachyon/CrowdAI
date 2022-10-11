using System.Collections.Generic;
using System.Linq;
using Agents;
using Managers;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Initializers
{
    public class Choke: IInitializer
    {
        private readonly Transform _ownObstacles;
        
        public Choke()
        {
            _ownObstacles = Manager.Instance.AllObstacles.Find("Choke");
        }
        
        public void PlaceAgents(Transform baseTransform, float size, List<Vector3> obstacles)
        {
            var zVal = Params.GroupSpawnScale;
            _ownObstacles.gameObject.SetActive(Params.EnableObstacles);
            
            var placedAgents = new List<Vector3>();
            var placedGoals = new List<Vector3>();
            var agentIdx = 0;

            var numAgents = baseTransform
                .Cast<Transform>()
                .Count(t => t.gameObject.activeInHierarchy);

            foreach (Transform agent in baseTransform)
            {
                if (!agent.gameObject.activeInHierarchy) continue;

                var agentBasic = agent.GetComponent<AgentBasic>();
                var goal = agentBasic.goal;

                Vector3 newPosition;
                Vector3 goalPosition;
                Vector3 goalOffset;
                Quaternion newRotation;
                
                float xMin, xMax, zMin, zMax;
                
                xMin = -9f;
                xMax = -9f + zVal * 2;
                zMin = -zVal;
                zMax = zVal;
                goalOffset = new Vector3(15f, 0, 0);
                newRotation = Quaternion.LookRotation(Vector3.right);
                if (Params.NiceColors) agentBasic.SetColor(Color.green, true);
                

                if (Params.GridSpawn)
                {
                    var gridSize = numAgents;
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

                if (Params.SharedGoal)
                {
                    goalPosition = goalOffset.normalized * 10f;
                    goal.localScale = new Vector3(4f, 1f, 4f);
                }
                else
                {
                    goalPosition = newPosition + goalOffset;
                    goal.localScale = agentBasic.goalScale;
                }
                
                agent.localPosition = newPosition;
                agent.localRotation = newRotation;
                goal.localPosition = goalPosition;

                // Save the placed agents
                placedAgents.Add(newPosition);
                placedGoals.Add(goalPosition);

                // Reset the dynamics
                agent.GetComponent<Rigidbody>().velocity = Vector3.zero;
                agent.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;

                // agentBasic.PreviousPosition = agent.localPosition;

                agentIdx++;
            }

        }

        public List<Vector3> GetObstacles()
        {
            return new List<Vector3>();
        }
    }
}