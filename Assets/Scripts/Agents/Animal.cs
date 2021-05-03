using System.Linq;
using Dynamics;
using Heuristics;
using Observers;
using Rewards;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Agents
{

    public enum AnimalType
    {
        Predator,
        Prey
    }
    public class Animal : Agent
    {
        private BufferSensorComponent _bufferSensor;
    
        protected Rigidbody Rigidbody;

        // public bool velocityControl = false;
        public float moveSpeed = 25f;
        public float rotationSpeed = 3f;
        public float dragFactor = 5f;

        public AnimalType type;
    
        public DynamicsEnum dynamicsType;
        private IDynamics _dynamics;

        public ObserversEnum observerType;
        private IObserver _observer;

        public RewardersEnum rewarderType;
        private IRewarder _rewarder;

        public HeuristicsEnum heuristicType;
        private IHeuristic _heuristic;
        
        public override void Initialize()
        {
            base.Initialize();
        
            Rigidbody = GetComponent<Rigidbody>();
            // startY = transform.localPosition.y;

            _dynamics = Dynamics.Mapper.GetDynamics(dynamicsType);
            _observer = Observers.Mapper.GetObserver(observerType);
            _rewarder = Rewards.Mapper.GetRewarder(rewarderType);
            _heuristic = Heuristics.Mapper.GetHeuristic(heuristicType);

            GetComponent<BehaviorParameters>().BrainParameters.VectorObservationSize = _observer.Size;
            
            _bufferSensor = GetComponent<BufferSensorComponent>();
            
            Physics.IgnoreLayerCollision(3, 3); // Ignore collisions between agents
        }
        
        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);
            // Debug.Log($"{name} OnAction at step {GetComponentInParent<Statistician>().Time}");
            _dynamics.ProcessActions(actions, Rigidbody, moveSpeed, rotationSpeed, dragFactor, 3f);
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            base.CollectObservations(sensor);

            _observer.Observe(sensor, transform);
            float reward = _rewarder.ComputeReward(transform);
            
            AddReward(reward);

            var layerMask = 1 << 3; // Only look at the Agent layer
            var nearbyObjects =
                Physics.OverlapSphere(transform.position, Params.SightRadius, layerMask)
                    .Where(c => c.CompareTag("Agent") & c.transform != transform) // Get only agents 
                    .OrderBy(c => Vector3.Distance(c.transform.localPosition, transform.localPosition))
                    .Select(c => MLUtils.GetPredatorPreyInfo(transform, c))
                    .Take(Params.SightAgents);
            
            foreach (var agentInfo in nearbyObjects)
            {
                // Debug.Log(String.Join(",", agentInfo));
                _bufferSensor.AppendObservation(agentInfo);
            }
            
            
        }


        public override void Heuristic(in ActionBuffers actionsOut)
        {
            // base.Heuristic(in actionsOut);

            _heuristic.DoAction(actionsOut, transform);
        }

        protected void OnCollisionEnter(Collision other)
        {

            AddReward(_rewarder.CollisionReward(transform, other, false));
        }

        public Transform FindNearestAgent(AnimalType otherType)
        {
            var closestAnimal = transform.parent.parent.GetComponentsInChildren<Animal>()
                .Where(a => a.type == otherType)
                .OrderBy(a => Vector3.Distance(a.transform.localPosition, transform.localPosition))
                .Select(a => a.transform)
                .First();

            return closestAnimal;
        }

        public float FindNearestDistance(AnimalType otherType)
        {
            var closestDistance = transform.parent.parent.GetComponentsInChildren<Animal>()
                .Where(a => a.type == otherType)
                .Select(a => Vector3.Distance(a.transform.localPosition, transform.localPosition))
                .Min();

            return closestDistance;
        }


    }
}