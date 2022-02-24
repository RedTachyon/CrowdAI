using System.Collections.Generic;
using Agents;
using UnityEngine;

namespace Initializers
{
    public class Circle : IInitializer
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

                float r = size / 2;
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
