using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Observers
{
    public interface IObserver
    {
        public void Observe(VectorSensor sensor, Transform transform, Transform goal);

        public int Size
        {
            get;
        }
    }

    public enum ObserversEnum
    {
        Absolute,
        Relative,
        RotRelative
    }
}