using System.Collections.Generic;
using System.Linq;
using Managers;
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

        public ActionSegment<float> LastAction;

        public bool CollectedGoal;
        
        public override void OnEpisodeBegin()
        {
            Debug.Log("Starting family");
            goalPosition = agents[0].Goal.localPosition;
            goalPosition = new Vector3(9, 0, 0);
            CollectedGoal = false;
            
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
            
            var currentDistance = MLUtils.FlatDistance(ownPosition, goal);
            var prevDistance = MLUtils.FlatDistance(PreviousPosition, goal);
            
            var reward = Params.Potential * (prevDistance - currentDistance);

            AddReward(reward);

            // Debug.Log($"Family reward: {reward}");
            
            PreviousPosition = ownPosition;
        }
        
        
        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);
            LastAction = actions.ContinuousActions;
            // Debug.Log($"Recording action at timestep {Manager.Instance.Timestep}");
            // Debug.Log("Action received family");
        }
        
        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var cActionsOut = actionsOut.ContinuousActions;
            cActionsOut[0] = 0;
            cActionsOut[1] = 1;
        }
        
        public Vector3 GetPosition()
        {
            // Returns the mean position of all agents

            var position = agents.Select(a => a.transform.position).Aggregate((a, b) => a + b) / agents.Count;
            return position;
        }
        
        
        public bool CheckGoal()
        {
            var ownPosition = GetPosition();
            var goal = goalPosition;
            var distance = MLUtils.FlatDistance(ownPosition, goal);
            return distance < Params.FamilyGoalRadius;
        }


        public void TryFinish()
        {
            if (CollectedGoal) return;
            
            var done = CheckGoal();
            if (!done) return;
            
            
            Debug.Log("Family done");
                
            foreach (AgentBasic agent in agents)
            {
                agent.CollectGoal();
            }

            AddReward(Params.Goal);
            CollectedGoal = true;

            // EndEpisode();
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