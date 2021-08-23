using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Observers
{
    public class AnimalProprioceptive : IObserver
    {
        public void Observe(VectorSensor sensor, Transform transform)
        {
            Vector3 position = transform.localPosition;
            Quaternion rotation = transform.localRotation;
            Vector3 velocity = transform.GetComponent<Rigidbody>().velocity;
            
            // Position: 2
            sensor.AddObservation(position.x / 10f);
            sensor.AddObservation(position.z / 10f);
        
            // Rotation: 1
            sensor.AddObservation(rotation.eulerAngles.y / 360f);
            
            // Velocity: 2, up to ~5
            sensor.AddObservation(velocity.x / 5f);
            sensor.AddObservation(velocity.z / 5f);
        }

        public int Size => 5;
    }
}