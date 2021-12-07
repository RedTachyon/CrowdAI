// using System.Collections.Generic;
// using System.Linq;
// using Agents;
// using UnityEngine;
//
// namespace Rewards
// {
//     public class AnimalChase : IRewarder
//     {
//         public float ComputeReward(Transform transform)
//         {
//             var type = transform.GetComponent<Animal>().type;
//             var isPredator = type == AnimalType.Predator;
//
//             var otherType = isPredator ? AnimalType.Predator : AnimalType.Prey;
//             
//             float factor = isPredator ? 1 : -1;
//
//             // var closestDistance = transform.parent.parent.GetComponentsInChildren<Animal>()
//             //     .Where(a => a.type != type)
//             //     .Select(a => Vector3.Distance(a.transform.localPosition, transform.localPosition))
//             //     .Min();
//
//             var closestDistance = transform.GetComponent<Animal>().FindNearestDistance(otherType);
//             
//             // Debug.Log($"Closest distance: {closestDistance}");
//             // transform.parent.parent
//
//             return factor * Params.PredatorPreyWeight * closestDistance;
//         }
//
//         public float CollisionReward(Transform transform, Collision other, bool stay)
//         {
//             return 0f;
//         }
//
//         public float TriggerReward(Transform transform, Collider other, bool stay)
//         {
//             return 0f;
//         }
//     }
// }
//
//
