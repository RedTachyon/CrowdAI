using Unity.MLAgents.Actuators;
using UnityEngine;

namespace Heuristics
{
    public interface IHeuristic
    {
        public void DoAction(in ActionBuffers actionsOut, Transform transform);
    }

    public enum HeuristicsEnum
    {
        Controls,
        Chase,
    }

    public class Mapper
    {
        public static IHeuristic GetHeuristic(HeuristicsEnum type)
        {
            IHeuristic heuristic = type switch
            {
                HeuristicsEnum.Controls => new Controls(),
                HeuristicsEnum.Chase => new Chase(),
                _ => null
            };

            return heuristic;
        }   
    }
}