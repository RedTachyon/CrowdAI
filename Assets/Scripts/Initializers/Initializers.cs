using UnityEditorInternal;
using UnityEngine;

namespace Initializers
{
    public interface IInitializer
    {
        public Vector3 GetPosition(int idx);
    }

    public enum InitializerEnum
    {
        Random,
        Circle,
        Hallway
    }
    
    public static class Mapper
    {
        public static IInitializer GetObserver(InitializerEnum initializerType)
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