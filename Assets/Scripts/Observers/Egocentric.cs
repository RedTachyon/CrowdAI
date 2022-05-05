using System;
using System.Linq;
using Agents;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Observers
{
    public class Egocentric : IObserver
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
            sensor.AddObservation(position.x / 10f);
            sensor.AddObservation(position.z / 10f);
        
            // Rotation: 1
            sensor.AddObservation(rotation.eulerAngles.y / 360f);
            
            // Relative position: 2
            var relPosition = Quaternion.Inverse(rotation) * (goalPosition - position);
            // var relPosition = goalPosition - position;
            sensor.AddObservation(relPosition.x / 20f);
            sensor.AddObservation(relPosition.z / 20f);

            // Debug.Log(relPosition);
            

            // Velocity: 2, up to ~5
            sensor.AddObservation(velocity.x / 5f);
            sensor.AddObservation(velocity.z / 5f);
            
            sensor.AddObservation(agent.CollectedGoal);
            sensor.AddObservation(agent.mass); // 8

        }
        public int Size => 9;

        public void ObserveAgents(BufferSensorComponent sensor, Transform transform)
        {
            LayerMask layerMask = 1 << LayerMask.NameToLayer("Agent");
            var nearbyObjects =
                Physics.OverlapSphere(transform.position, Params.SightRadius, layerMask)
                    .Where(c => c.CompareTag("Agent") && c.transform != transform) // Get only agents 
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

        public static float[] GetColliderInfo(Transform baseTransform, Collider collider)
        {
            
            var rigidbody = collider.GetComponent<Rigidbody>();
            var transform = collider.transform;

            var pos = transform.localPosition;
            var velocity = rigidbody.velocity;

            var rotation = baseTransform.localRotation;
            pos = Quaternion.Inverse(rotation) * (pos - baseTransform.localPosition);
            velocity = Quaternion.Inverse(rotation) * velocity;
        
            
        
            return new[] {pos.x, pos.z, velocity.x, velocity.z};
        }
    }
}