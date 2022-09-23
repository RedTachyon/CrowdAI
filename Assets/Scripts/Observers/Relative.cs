using System;
using System.Collections.Generic;
using System.Linq;
using Agents;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Observers
{
    public class Relative : IObserver
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

        
            Vector3 position = transform.localPosition;
            Quaternion rotation = transform.localRotation;
            Vector3 velocity = transform.GetComponent<Rigidbody>().velocity;
            Vector3 goalPosition = goal.localPosition;
            
            // Position: 2
            sensor.AddObservation(position.x / 10f);  // 0
            sensor.AddObservation(position.z / 10f);  // 1
        
            // Rotation: 1
            sensor.AddObservation(rotation.eulerAngles.y / 360f);  // 2

            // Relative position: 2
            var relPosition = goalPosition - position;
            sensor.AddObservation(relPosition.x / 10f);  // 3
            sensor.AddObservation(relPosition.z / 10f);  // 4
            

            // Velocity: 2, up to ~2
            sensor.AddObservation(velocity.x / 2f);  // 5
            sensor.AddObservation(velocity.z / 2f);  // 6
            
            sensor.AddObservation(agent.CollectedGoal); // 7
            sensor.AddObservation(agent.mass); // 8


        }
        public int Size => 9;

        public IEnumerable<string> ObserveAgents(BufferSensorComponent sensor, Transform transform, bool useAcceleration)
        {
            
            // Collect Buffer observations
            LayerMask layerMask = 1 << LayerMask.NameToLayer("Agent");
            var nearbyColliders =
                Physics.OverlapSphere(transform.position, Params.SightRadius, layerMask)
                    .Where(c => c.CompareTag("Agent") && c.transform != transform) // Get only agents
                    .Where(c => MLUtils.Visible(transform, c.transform, Params.MinCosine)) // Cone of vision
                    .OrderBy(c => Vector3.Distance(c.transform.localPosition, transform.localPosition))
                    .ToList();
                
            var names = nearbyColliders
                .Select(c => c.transform.name)
                .Take(Params.SightAgents);
            
            var nearbyObjects = nearbyColliders
                .Select(c => GetColliderInfo(transform, c, useAcceleration))
                .Take(Params.SightAgents);
        
            // Debug.Log(nearbyObjects);
            foreach (var agentInfo in nearbyObjects)
            {
                // Debug.Log(String.Join(",", agentInfo));
                sensor.AppendObservation(agentInfo);
            }

            return names;
        }

        private static float[] GetColliderInfo(Transform baseTransform, Collider collider, bool useAcceleration)
        {
            
            var rigidbody = collider.GetComponent<Rigidbody>();
            var agent = collider.GetComponent<AgentBasic>();
            var transform = collider.transform;
            
            var pos = transform.localPosition - baseTransform.localPosition;
            var velocity = rigidbody.velocity;

            float[] obs;
            if (useAcceleration)
            {
                var acceleration = agent == null
                    ? Vector3.zero
                    : velocity - agent.PreviousVelocity;

                obs = new[] {pos.x, pos.z, velocity.x, velocity.z, agent.mass, acceleration.x, acceleration.z};
            }
            else
            {
                obs = new[] {pos.x, pos.z, velocity.x, velocity.z, agent.mass};
            }


            return obs;
        }
    }
}