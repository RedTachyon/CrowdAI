using System.Collections.Generic;
using UnityEngine;

namespace Initializers
{
    public interface IInitializer
    {
        public void PlaceAgents(Transform baseTransform, float size, List<Vector3> obstacles);
        public List<Vector3> GetObstacles();
    }

    public enum InitializerEnum
    {
        Random,
        Circle,
        CircleBlock,
        Crossway,
        Corridor,
        JsonInitializer,
    }
    
    public static class Mapper
    {
        // Cache the dataless initializers
        private static IInitializer _random = new Random();
        private static IInitializer _circle = new Circle();
        private static IInitializer _circleBlock = new CircleBlock();
        private static IInitializer _crossway = new Crossway();
        private static IInitializer _corridor = new Corridor();
        
        public static IInitializer GetInitializer(InitializerEnum initializerType, string path = null)
        {
            IInitializer initializer = initializerType switch
            {
                InitializerEnum.Random => _random,
                InitializerEnum.Circle => _circle,
                InitializerEnum.CircleBlock => _circleBlock,
                InitializerEnum.Crossway => _crossway,
                InitializerEnum.Corridor => _corridor,
                InitializerEnum.JsonInitializer => new JsonInitializer(path),
                _ => null
            };

            return initializer;
        }

    }
}