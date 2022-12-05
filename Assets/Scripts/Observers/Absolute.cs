using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Agents;
using Managers;
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
            var goal = agent.Goal;
        
            var position = transform.localPosition;
            var rotation = transform.localRotation;
            var velocity = transform.GetComponent<Rigidbody>().velocity;
            var goalPosition = goal.localPosition;
            
            // Position: 2
            sensor.AddObservation(position.x / 10f); // 1
            sensor.AddObservation(position.z / 10f); // 2
        
            // Rotation: 1
            sensor.AddObservation(Mathf.Deg2Rad * rotation.eulerAngles.y); // 3
        
            // Goal position: 2
            sensor.AddObservation(goalPosition.x / 10f); // 4
            sensor.AddObservation(goalPosition.z / 10f); // 5

            // Velocity: 2, up to ~2
            sensor.AddObservation(velocity.x / 2f); // 6
            sensor.AddObservation(velocity.z / 2f); // 7
            
            sensor.AddObservation(agent.CollectedGoal); // 8
            sensor.AddObservation(agent.mass); // 9
            
            sensor.AddObservation(agent.e_s); // 10
            sensor.AddObservation(agent.e_w); // 11

            sensor.AddObservation(agent.PreferredSpeed); // 12
            
            sensor.AddObservation(Manager.Instance.NormedTime); // 13
        }
        public int Size => 13;



        public float[] GetColliderInfo(Transform baseTransform, Collider collider)
        {
            var rigidbody = collider.GetComponent<Rigidbody>();
            var agent = collider.GetComponent<AgentBasic>();

            var transform = collider.transform;
        
            var pos = transform.localPosition;
            var velocity = rigidbody.velocity;

            float[] obs;

            obs = new[] {pos.x, pos.z, velocity.x, velocity.z, agent.mass};


            return obs;
        }
    }

}