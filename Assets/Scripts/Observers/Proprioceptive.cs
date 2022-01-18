using Agents;
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
            sensor.AddObservation(position.x / 10f);
            sensor.AddObservation(position.z / 10f);
        
            // Rotation: 1
            sensor.AddObservation(rotation.eulerAngles.y / 360f);
            
            // Velocity: 2, up to ~5
            sensor.AddObservation(velocity.x / 5f);
            sensor.AddObservation(velocity.z / 5f);
            
            sensor.AddObservation(agent.CollectedGoal);
            sensor.AddObservation(agent.mass); // 8

        }

        public void ObserveAgents(BufferSensorComponent sensor, Transform transform)
        {
            
        }

        public int Size => 7;
    }
}