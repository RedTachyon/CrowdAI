using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Observers
{
    public interface IObserver
    {
        public void Observe(VectorSensor sensor, Transform transform);

        public List<Transform> ObserveAgents(BufferSensorComponent sensor, Transform transform, bool useAcceleration)
        {
            // Collect Buffer observations
            LayerMask layerMask = 1 << LayerMask.NameToLayer("Agent");
            
            var nearbyColliders =
                Physics.OverlapSphere(transform.position, Params.SightRadius, layerMask)
                    .Where(c => c.CompareTag("Agent") && c.transform != transform) // Get only agents
                    .Where(c => MLUtils.Visible(transform, c.transform, Params.MinCosine)) // Cone of vision
                    .OrderBy(c => Vector3.Distance(c.transform.localPosition, transform.localPosition))
                    .ToList();

            var agents = nearbyColliders
                .Select(c => c.transform)
                .Take(Params.SightAgents)
                .ToList();
            
            var nearbyObjects = nearbyColliders
                .Select(c => GetColliderInfo(transform, c, useAcceleration))
                .Take(Params.SightAgents);
        
            // Debug.Log(nearbyObjects);
            foreach (var agentInfo in nearbyObjects)
            {
                // Debug.Log(String.Join(",", agentInfo));
                sensor.AppendObservation(agentInfo);
            }

            return agents;
        }

        public float[] GetColliderInfo(Transform baseTransform, Collider collider, bool useAcceleration);

        public int Size
        {
            get;
        }
    }

    public enum ObserversEnum
    {
        Absolute,
        Relative,
        Egocentric,
        Proprioceptive,
    }
    
    public static class Mapper
    {
        public static IObserver GetObserver(ObserversEnum obsType)
        {
            IObserver observer = obsType switch
            {
                ObserversEnum.Absolute => new Absolute(),
                ObserversEnum.Relative => new Relative(),
                ObserversEnum.Egocentric => new Egocentric(),
                ObserversEnum.Proprioceptive => new Proprioceptive(),
                _ => null
            };

            return observer;
        }
    }
}