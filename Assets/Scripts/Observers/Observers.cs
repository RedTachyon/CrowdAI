using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Observers
{
    public interface IObserver
    {
        public void Observe(VectorSensor sensor, Transform transform);

        public void ObserveAgents(BufferSensorComponent sensor, Transform transform, bool useAcceleration);

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