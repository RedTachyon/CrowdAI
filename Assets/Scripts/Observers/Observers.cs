using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Observers
{
    public interface IObserver
    {
        public void Observe(VectorSensor sensor, Transform transform);

        public int Size
        {
            get;
        }
    }

    public enum ObserversEnum
    {
        Absolute,
        Relative,
        RotRelative,
        Proprioceptive
    }
    
    public static class Mapper
    {
        public static IObserver GetObserver(ObserversEnum obsType)
        {
            IObserver observer = obsType switch
            {
                ObserversEnum.Absolute => new Absolute(),
                ObserversEnum.Relative => new Relative(),
                ObserversEnum.RotRelative => new RotRelative(),
                ObserversEnum.Proprioceptive => new Proprioceptive(),
                _ => null
            };

            return observer;
        }
    }
}