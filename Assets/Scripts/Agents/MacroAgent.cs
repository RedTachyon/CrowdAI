using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Agents
{
    public class MacroAgent : Agent
    {
        public List<AgentBasic> agents;
        
        public override void OnEpisodeBegin()
        {
            Debug.Log("Starting family");
        }


        public override void CollectObservations(VectorSensor sensor)
        {
            base.CollectObservations(sensor);
            Debug.Log("Collecting observations family");
        }
        
        
        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);
            Debug.Log("Action received family");
        }
    }
}