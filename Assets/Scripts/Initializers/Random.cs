using System.Collections.Generic;
using Agents;
using UnityEngine;

namespace Initializers
{
    public class Random : IInitializer
    {
        public void PlaceAgents(Transform baseTransform, float size, List<Vector3> obstacles)
        {
            var placedAgents = new List<Vector3>();
            var placedGoals = new List<Vector3>();

            foreach (Transform agent in baseTransform)
            {
                var goal = agent.GetComponent<AgentBasic>().goal;

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
            
                // Save the placed agents
                placedAgents.Add(newPosition);
                placedGoals.Add(goalPosition);

                // Reset the dynamics
                agent.GetComponent<Rigidbody>().velocity = Vector3.zero;
                agent.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        
                agent.GetComponent<AgentBasic>().PreviousPosition = agent.localPosition;
            }
            
        }
    }
}