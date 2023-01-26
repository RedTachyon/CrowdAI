using System.Collections.Generic;
using System.Linq;
using Agents;
using Managers;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Initializers
{
    public class Family : IInitializer
    {
        private readonly Transform _ownObstacles;
        
        public Family()
        {
            _ownObstacles = Manager.Instance.AllObstacles.Find("Family");
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
            
            // Spawn around (-9, _, 0), goal around (+9, _, 0)

            foreach (Transform agent in baseTransform)
            {
                if (!agent.gameObject.activeInHierarchy) continue;

                var agentBasic = agent.GetComponent<AgentBasic>();
                var goal = agentBasic.Goal;

                Vector3 newPosition;
                Vector3 goalPosition;
                Vector3 goalOffset;
                Quaternion newRotation;
                
                // float xMin, xMax, zMin, zMax;

                var newX = -6 + UnityEngine.Random.Range(-5, 5);
                var newZ = 0 + UnityEngine.Random.Range(-5, 5);
                
                newPosition = new Vector3(newX, agent.localPosition.y, newZ);
                newRotation = Quaternion.LookRotation(Vector3.right);
                goalPosition = new Vector3(9, agent.localPosition.y, 0);
                
                
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