using System;
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
            sensor.AddObservation(relPosition.x / 20f);  // 3
            sensor.AddObservation(relPosition.z / 20f);  // 4
            

            // Velocity: 2, up to ~5
            sensor.AddObservation(velocity.x / 5f);  // 5
            sensor.AddObservation(velocity.z / 5f);  // 6
            
            sensor.AddObservation(agent.CollectedGoal); // 7

        }
        public int Size => 8;

        public void ObserveAgents(BufferSensorComponent sensor, Transform transform)
        {
            
            // Collect Buffer observations
            const int layerMask = 1 << 3; // Only look at the Agent layer
            var nearbyObjects =
                Physics.OverlapSphere(transform.position, Params.SightRadius, layerMask)
                    .Where(c => c.CompareTag("Agent") & c.transform != transform) // Get only agents 
                    .OrderBy(c => Vector3.Distance(c.transform.localPosition, transform.localPosition))
                    .Select(c => GetColliderInfo(transform, c))
                    .Take(Params.SightAgents);
        
            // Debug.Log(nearbyObjects);
            foreach (var agentInfo in nearbyObjects)
            {
                // Debug.Log(String.Join(",", agentInfo));
                sensor.AppendObservation(agentInfo);
            }
        }

        private static float[] GetColliderInfo(Transform baseTransform, Collider collider)
        {
            
            var rigidbody = collider.GetComponent<Rigidbody>();
            var transform = collider.transform;
            
            var pos = transform.localPosition - baseTransform.localPosition;
            var velocity = rigidbody.velocity;
            
            return new[] {pos.x, pos.z, velocity.x, velocity.z};
        }
    }
}