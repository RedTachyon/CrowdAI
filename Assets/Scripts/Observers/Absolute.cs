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
            sensor.AddObservation(position.x / 10f);
            sensor.AddObservation(position.z / 10f);
        
            // Rotation: 1
            sensor.AddObservation(rotation.eulerAngles.y / 360f);
        
            // Goal position: 2
            sensor.AddObservation(goalPosition.x / 10f);
            sensor.AddObservation(goalPosition.z / 10f);

            // Velocity: 2, up to ~5
            sensor.AddObservation(velocity.x / 5f);
            sensor.AddObservation(velocity.z / 5f);
        }

        public int Size => 7;
    }

}