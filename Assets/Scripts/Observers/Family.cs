using Agents;
using Managers;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Observers
{
    public class Family : IObserver
    {
        public void Observe(VectorSensor sensor, Transform transform)
        {
            // Identical as egocentric for now, but no goal?
            
            
            var agent = transform.GetComponent<AgentBasic>();
            var goal = agent.Goal;

        
            Vector3 position = transform.localPosition;
            Quaternion rotation = transform.localRotation;
            Vector3 velocity = transform.GetComponent<Rigidbody>().velocity;
            Vector3 goalPosition = goal.localPosition;
            
            // Position: 2
            sensor.AddObservation(position.x / 10f); // 1
            sensor.AddObservation(position.z / 10f); // 2
        
            // Rotation: 1
            sensor.AddObservation(rotation.eulerAngles.y / 360f); // 3
            
            // Relative position: 2
            // var relPosition = Quaternion.Inverse(rotation) * (goalPosition - agent.Family.GetPosition());
            // // var relPosition = goalPosition - position;
            // sensor.AddObservation(relPosition.x / 20f); // 4
            // sensor.AddObservation(relPosition.z / 20f); // 5

            // Debug.Log(relPosition);
            

            // Velocity: 2, up to ~5
            sensor.AddObservation(velocity.x / 5f); // 6
            sensor.AddObservation(velocity.z / 5f); // 7
            
            sensor.AddObservation(agent.CollectedGoal); // 8
            sensor.AddObservation(agent.mass); // 9

            sensor.AddObservation(agent.e_s); // 10
            sensor.AddObservation(agent.e_w); // 11
            
            sensor.AddObservation(agent.PreferredSpeed); // 12
            sensor.AddObservation(Manager.Instance.NormedTime); // 13
        }
        
        public int Size => 11;

        public float[] GetColliderInfo(Transform baseTransform, Collider collider)
        {
            
            var rigidbody = collider.GetComponent<Rigidbody>();
            var transform = collider.transform;

            var agent = collider.GetComponent<AgentBasic>();

            var pos = transform.localPosition;
            var velocity = rigidbody.velocity;

            var rotation = baseTransform.localRotation;
            pos = Quaternion.Inverse(rotation) * (pos - baseTransform.localPosition);
            velocity = Quaternion.Inverse(rotation) * velocity;

            var mass = agent == null ? 0f : agent.mass;
            
            float[] obs;

  
            obs = new[] { pos.x, pos.z, velocity.x, velocity.z, mass};

            // if (baseTransform.name == "Person")
            // {
            //     Debug.Log($"{baseTransform.name} sees {transform.name} at {pos} with velocity {velocity} and mass {agent.mass}");
            // }

            return obs;
        }
    }
}