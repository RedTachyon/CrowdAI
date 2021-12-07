using System.Collections.Generic;
using Agents;
using UnityEngine;

namespace Initializers
{
    public class Circle : IInitializer
    {        
        public void PlaceAgents(Transform baseTransform)
        {
            var placedAgents = new List<Vector3>();
            var placedGoals = new List<Vector3>();
            var agentIdx = 0;

            var numAgents = baseTransform.childCount;

            foreach (Transform agent in baseTransform)
            {  
                var goal = agent.GetComponent<AgentBasic>().goal;

                const float r = 9;
                var x = r * Mathf.Cos((float) agentIdx / numAgents * Constants.Tau);
                var z = r * Mathf.Sin((float) agentIdx / numAgents * Constants.Tau);
                var newPosition = new Vector3(x, agent.localPosition.y, z);
                var goalPosition = new Vector3(-x, goal.localPosition.y, -z);
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
