using System;
using System.Drawing;
using System.Linq;
using Agents;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Rendering;

namespace Observers
{
    public class Absolute : IObserver
    {
        public void Observe(VectorSensor sensor, Transform transform)
        {
            // Debug.Log($"{name} CollectObs at step {GetComponentInParent<Statistician>().Time}");
        
            // RayPerceptionSensor structure:
            // 0 - n_tags: one-hot encoding of what was hit
            // n_tags: whether *something* was hit
            // n_tags + 1: normalized distance

            var agent = transform.GetComponent<AgentBasic>();
            var goal = agent.goal;
        
            var position = transform.localPosition;
            var rotation = transform.localRotation;
            var velocity = transform.GetComponent<Rigidbody>().velocity;
            var goalPosition = goal.localPosition;
            
            // Position: 2
            sensor.AddObservation(position.x / 10f); // 0
            sensor.AddObservation(position.z / 10f); // 1
        
            // Rotation: 1
            sensor.AddObservation(Mathf.Deg2Rad * rotation.eulerAngles.y); // 2
        
            // Goal position: 2
            sensor.AddObservation(goalPosition.x / 10f); // 3
            sensor.AddObservation(goalPosition.z / 10f); // 4

            // Velocity: 2, up to ~2
            sensor.AddObservation(velocity.x / 2f); // 5
            sensor.AddObservation(velocity.z / 2f); // 6
            
            sensor.AddObservation(agent.CollectedGoal); // 7
            sensor.AddObservation(agent.mass); // 8
        }
        public int Size => 9;

        public void ObserveAgents(BufferSensorComponent sensor, Transform transform)
        {
            // Collect Buffer observations
            LayerMask layerMask = 1 << LayerMask.NameToLayer("Agent");
            
            var nearbyObjects =
                Physics.OverlapSphere(transform.position, Params.SightRadius, layerMask)
                    .Where(c => c.CompareTag("Agent") && c.transform != transform) // Get only agents
                    .Where(c => MLUtils.Visible(transform, c.transform, Params.MaxCosine)) // Cone of vision
                    .OrderBy(c => Vector3.Distance(c.transform.localPosition, transform.localPosition))
                    .Select(GetColliderInfo)
                    .Take(Params.SightAgents);
        
            // Debug.Log(nearbyObjects);
            foreach (var agentInfo in nearbyObjects)
            {
                // Debug.Log(String.Join(",", agentInfo));
                sensor.AppendObservation(agentInfo);
            }
        }

        private static float[] GetColliderInfo(Collider collider)
        {
            var rigidbody = collider.GetComponent<Rigidbody>();
            var agent = collider.GetComponent<AgentBasic>();
            var transform = collider.transform;
        
            var pos = transform.localPosition;
            var velocity = rigidbody.velocity;

            var acceleration = velocity - agent.PreviousVelocity;
            // TODO: add acceleration info, make it switchable
            
            return new[] {pos.x, pos.z, velocity.x, velocity.z};
        }
    }

}