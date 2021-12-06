using System.Collections.Generic;
using UnityEngine;

namespace Initializers
{
    public interface IInitializer
    {
        public void PlaceAgents(Transform baseTransform);
    }

    public enum InitializerEnum
    {
        Random,
        Circle,
        Hallway
    }
    
    public static class Mapper
    {
        public static IInitializer GetInitializer(InitializerEnum initializerType)
        {
            IInitializer initializer = initializerType switch
            {
                InitializerEnum.Random => new Random(),
                InitializerEnum.Circle => new Circle(),
                InitializerEnum.Hallway => new Hallway(),
                _ => null
            };

            return initializer;
        }
    }
}