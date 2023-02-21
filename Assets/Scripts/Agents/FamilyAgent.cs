using System.Collections.Generic;
using System.Linq;
using Observers;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Agents
{
    public class FamilyAgent : Agent
    {
        [SerializeField] private List<AgentBasic> agents;
        
        public Vector3 goalPosition;

        private Vector3 PreviousPosition;
        
        public override void OnEpisodeBegin()
        {
            Debug.Log("Starting family");
            goalPosition = agents[0].Goal.localPosition;
            goalPosition = new Vector3(9, 0, 0);
            PreviousPosition = GetPosition();
        }


        public override void CollectObservations(VectorSensor sensor)
        {
            base.CollectObservations(sensor);

            // Debug.Log("Collecting observations family");
            
            var ownPosition = GetPosition();
            var goal = goalPosition;
            
            
            // Observations:
            
            sensor.AddObservation(ownPosition.x);
            sensor.AddObservation(ownPosition.z);
            sensor.AddObservation(goal.x);
            sensor.AddObservation(goal.z);
        }
        
        
        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);
            // Debug.Log("Action received family");
        }
        
        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var cActionsOut = actionsOut.ContinuousActions;
            cActionsOut[0] = 0;
            cActionsOut[1] = 0;
        }
        
        public Vector3 GetPosition()
        {
            // Returns the mean position of all agents
            // TODO: cache this value

            var position = agents.Select(a => a.transform.position).Aggregate((a, b) => a + b) / agents.Count;
            return position;
        }
        
        
        
        public void AddAgent(AgentBasic agent)
        {
            agents.Add(agent);
            agent.Family = this;
        }
        
        public void RemoveAgent(AgentBasic agent)
        {
            agents.Remove(agent);
            agent.Family = null;
        }
        
        public void ResetAgents()
        {
            foreach (var agent in agents)
            {
                agent.Family = null;
            }
            agents.Clear();
        }
    }
}