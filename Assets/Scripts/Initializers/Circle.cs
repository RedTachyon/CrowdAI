using System.Collections.Generic;
using System.Linq;
using Agents;
using Managers;
using UnityEngine;

namespace Initializers
{
    public class Circle : IInitializer
    {        
        private readonly Transform _ownObstacles;
        private readonly List<Vector3> _obstaclePositions;

        public Circle()
        {
            _ownObstacles = Manager.Instance.AllObstacles.Find("Circle");
            _obstaclePositions = _ownObstacles.Cast<Transform>().Select(obstacle => obstacle.transform.position).ToList();
        }
        public void PlaceAgents(Transform baseTransform, float size, List<Vector3> obstacles)
        {
            _ownObstacles.gameObject.SetActive(Params.EnableObstacles);

            var placedAgents = new List<Vector3>();
            var placedGoals = new List<Vector3>();
            var agentIdx = 0;

            var numAgents = baseTransform
                .Cast<Transform>()
                .Count(t => t.gameObject.activeInHierarchy);
            
            foreach (Transform agent in baseTransform)
            {
                var agentBasic = agent.GetComponent<AgentBasic>();
                var goal = agentBasic.goal;

                float r = size;
                var x = r * Mathf.Cos((float) agentIdx / numAgents * Constants.Tau);
                var z = r * Mathf.Sin((float) agentIdx / numAgents * Constants.Tau);
                var spawnNoise = MLUtils.GetNoise(Params.SpawnNoiseScale);
                var goalNoise = MLUtils.GetNoise(Params.SpawnNoiseScale);
                var newPosition = new Vector3(x, agent.localPosition.y, z) + spawnNoise;
                var goalPosition = new Vector3(-x, goal.localPosition.y, -z) + goalNoise;
                var newRotation = Quaternion.LookRotation(goalPosition, Vector3.up);

                agent.localPosition = newPosition;
                agent.localRotation = newRotation;
                goal.localPosition = goalPosition;
                goal.localScale = Manager.Instance.goalScale;
            
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

        public List<Vector3> GetObstacles()
        {
            return _obstaclePositions;
        }
    }
        
}
