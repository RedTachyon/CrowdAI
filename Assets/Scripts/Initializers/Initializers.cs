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
        Choke,
        Family,
        Car,
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
        private static IInitializer _choke = new Choke();
        private static IInitializer _family = new Family();
        private static IInitializer _car = new Car();
        
        public static IInitializer GetInitializer(InitializerEnum initializerType, string path = null)
        {
            IInitializer initializer = initializerType switch
            {
                InitializerEnum.Random => _random,
                InitializerEnum.Circle => _circle,
                InitializerEnum.CircleBlock => _circleBlock,
                InitializerEnum.Crossway => _crossway,
                InitializerEnum.Corridor => _corridor,
                InitializerEnum.Choke => _choke,
                InitializerEnum.Family => _family,
                InitializerEnum.Car => _car,
                InitializerEnum.JsonInitializer => new JsonInitializer(path),
                _ => null
            };

            return initializer;
        }

    }
}