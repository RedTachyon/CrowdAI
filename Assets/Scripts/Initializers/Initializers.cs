using System.Collections.Generic;
using UnityEngine;

namespace Initializers
{
    public interface IInitializer
    {
        public void PlaceAgents(Transform baseTransform, float size, List<Vector3> obstacles);
    }

    public enum InitializerEnum
    {
        Random,
        Circle,
        Hallway,
        JsonInitializer,
    }
    
    public static class Mapper
    {
        public static IInitializer GetInitializer(InitializerEnum initializerType, string path = null)
        {
            IInitializer initializer = initializerType switch
            {
                InitializerEnum.Random => new Random(),
                InitializerEnum.Circle => new Circle(),
                InitializerEnum.Hallway => new Hallway(),
                InitializerEnum.JsonInitializer => new JsonInitializer(path),
                _ => null
            };

            return initializer;
        }

    }
}