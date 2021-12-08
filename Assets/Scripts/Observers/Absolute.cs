using System;
using System.Drawing;
using Agents;
using Unity.MLAgents.Sensors;
using UnityEngine;

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

            Transform goal = transform.GetComponent<AgentBasic>().goal;
        
            Vector3 position = transform.localPosition;
            Quaternion rotation = transform.localRotation;
            Vector3 velocity = transform.GetComponent<Rigidbody>().velocity;
            Vector3 goalPosition = goal.localPosition;
            
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
        }

        public int Size => 7;
    }

}