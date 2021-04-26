using System;

namespace Observers
{
    public static class Mapper
    {
        public static IObserver GetObserver(ObserversEnum obsType)
        {
            IObserver observer = obsType switch
            {
                ObserversEnum.Absolute => new Absolute(),
                ObserversEnum.Relative => new Relative(),
                ObserversEnum.RotRelative => new RotRelative(),
                _ => null
            };

            return observer;
        }
    }
}