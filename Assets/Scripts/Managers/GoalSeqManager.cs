using System.Collections.Generic;
using Agents;
using Unity.MLAgents;
using UnityEngine;

namespace Managers
{
    public class GoalSeqManager : Manager
    {

        private List<Vector3> _obstaclePositions = new();

        public new void Awake()
        {
            base.Awake();
            // TODO: Make recursive
            foreach (Transform obs in obstacles)
            {
                _obstaclePositions.Add(obs.localPosition);
                Debug.Log(obs.name);
            }
            
        }
        public override void ReachGoal(Agent agent)
        {
            Debug.Log("Relocating the goal");
            var goal = agent.GetComponent<AgentBasic>().goal;
            var goalPosition = MLUtils.NoncollidingPosition(
                -4f,
                4f,
                -4f,
                4f,
                goal.localPosition.y,
                _obstaclePositions
                );
            goal.localPosition = goalPosition;

        }
    }
}