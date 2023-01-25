using System.Collections.Generic;
using Agents;
using Managers;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Observers
{
    public class Proprioceptive : IObserver
    {
        public void Observe(VectorSensor sensor, Transform transform)
        {
            Vector3 position = transform.localPosition;
            Quaternion rotation = transform.localRotation;
            Vector3 velocity = transform.GetComponent<Rigidbody>().velocity;
            var agent = transform.GetComponent<AgentBasic>();

            
            // Position: 2
            sensor.AddObservation(position.x / 10f); // 1
            sensor.AddObservation(position.z / 10f); // 2
        
            // Rotation: 1
            sensor.AddObservation(rotation.eulerAngles.y / 360f); // 3
            
            // Velocity: 2, up to ~5
            sensor.AddObservation(velocity.x / 5f); // 4
            sensor.AddObservation(velocity.z / 5f); // 5
            
            sensor.AddObservation(agent.CollectedGoal); // 6
            sensor.AddObservation(agent.mass); // 7
            
            sensor.AddObservation((float) Manager.Instance.DecisionTimestep / Manager.Instance.maxStep); // 8


        }
        
        public int Size => 8;
        public IEnumerable<string> ObserveAgents(BufferSensorComponent sensor, Transform transform, bool useAcceleration)
        {
            return new List<string>();
        }

        public float[] GetColliderInfo(Transform baseTransform, Collider collider, bool useAcceleration)
        {
            return new float[] { };
        }

    }
}