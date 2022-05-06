using System.Collections.Generic;
using System.Linq;
using Agents;
using Managers;
using UnityEngine;

namespace Initializers
{
    public class Random : IInitializer
    {
        private readonly Transform _ownObstacles;
        private readonly List<Vector3> _obstaclePositions;

        public Random()
        {
            _ownObstacles = Manager.Instance.AllObstacles.Find("Random");
            _obstaclePositions = _ownObstacles.Cast<Transform>().Select(obstacle => obstacle.transform.position).ToList();
        }
        
        public void PlaceAgents(Transform baseTransform, float size, List<Vector3> obstacles)
        {
            _ownObstacles.gameObject.SetActive(Params.EnableObstacles);
            var placedAgents = new List<Vector3>(obstacles);
            var placedGoals = new List<Vector3>(obstacles);

            foreach (Transform agent in baseTransform)
            {
                // Debug.Log($"Forbidden positions: {placedAgents.Count}");
                var agentBasic = agent.GetComponent<AgentBasic>();
                var goal = agentBasic.goal;
                
                var newPosition = MLUtils.NoncollidingPosition(
                    -size,
                    size,
                    -size,
                    size,
                    agent.localPosition.y,
                    placedAgents);

                var goalPosition = MLUtils.NoncollidingPosition(
                    -size,
                    size,
                    -size,
                    size,
                    goal.localPosition.y,
                    placedGoals);

                var newRotation = Quaternion.Euler(0f, UnityEngine.Random.Range(0f, 360f), 0f);
                
                agent.localPosition = newPosition;
                agent.localRotation = newRotation;
                goal.localPosition = goalPosition;
                goal.localScale = agentBasic.goalScale;
            
                // Save the placed agents
                placedAgents.Add(newPosition);
                placedGoals.Add(goalPosition);

                // Reset the dynamics
                agent.GetComponent<Rigidbody>().velocity = Vector3.zero;
                agent.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        
                agent.GetComponent<AgentBasic>().PreviousPosition = agent.localPosition;
            }
            
        }

        public List<Vector3> GetObstacles()
        {
            return _obstaclePositions;
        }
    }
}