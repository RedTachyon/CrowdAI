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

                // Vector3 newPosition;
                // Vector3 goalPosition;
                // Vector3 goalOffset;
                // Quaternion newRotation;
                //
                //
                // var u1 = UnityEngine.Random.value;
                // var u2 = UnityEngine.Random.value;
                //
                // var newX = -9 + 1 * Mathf.Sqrt(-2 * Mathf.Log(u1)) * Mathf.Cos(2 * Mathf.PI * u2);
                // var newZ = 0 + 1 * Mathf.Sqrt(-2 * Mathf.Log(u1)) * Mathf.Sin(2 * Mathf.PI * u2);
                
                var newPosition = MLUtils.NoncollidingPosition(
                    -11,
                    -7,
                    -2,
                    2,
                    agent.localPosition.y,
                    placedAgents);
                
                // newPosition = new Vector3(newX, agent.localPosition.y, newZ);
                var newRotation = Quaternion.LookRotation(Vector3.right);
                var goalPosition = new Vector3(9, agent.localPosition.y, 0);
                
                
                agent.localPosition = newPosition;
                agent.localRotation = newRotation;
                goal.localPosition = goalPosition;

                var scale = 2 * Params.FamilyGoalRadius;
                goal.localScale = new Vector3(scale, goal.localScale.y, scale);

                // Save the placed agents
                placedAgents.Add(newPosition);
                placedGoals.Add(goalPosition);

                // Reset the dynamics
                agent.GetComponent<Rigidbody>().velocity = Vector3.zero;
                agent.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;

                // agentBasic.PreviousPosition = agent.localPosition;
                
                // Disable collider for the goal
                agentBasic.Goal.GetComponent<Collider>().enabled = false;

                agentIdx++;
            }

        }

        public List<Vector3> GetObstacles()
        {
            return new List<Vector3>();
        }
    }
}