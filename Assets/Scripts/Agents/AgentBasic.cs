using System.Linq;
using Dynamics;
using Managers;
using Observers;
using Rewards;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;

// Proposed reward structure:
// 16.5 total reward for approaching the goal
// 0.1 reward per decision step for reaching the goal (10 reward per 100 steps, max ~40)
// -0.01 reward per decision step for collisions (-1 reward per 100 steps)
// -0.01 reward per decision step

namespace Agents
{
    public class AgentBasic : Agent
    {
    
        private Material _material;
        private Color _originalColor;
        internal bool CollectedGoal;

        private BufferSensorComponent _bufferSensor;
    
        protected Rigidbody Rigidbody;

        // public bool velocityControl = false;
        public float moveSpeed = 25f;
        public float rotationSpeed = 3f;
        public float dragFactor = 5f;

        public DynamicsEnum dynamicsType;
        private IDynamics _dynamics;

        public ObserversEnum observerType;
        private IObserver _observer;

        public RewardersEnum rewarderType;
        private IRewarder _rewarder;


        protected int Unfrozen = 1;

        internal int Collision = 0;


        public Transform goal;


        [HideInInspector] public Vector3 startPosition;

        [HideInInspector] public Quaternion startRotation;
    
        // public Transform goal;


        public override void Initialize()
        {
            base.Initialize();
        
            Rigidbody = GetComponent<Rigidbody>();
            // startY = transform.localPosition.y;
            startPosition = transform.localPosition;
            startRotation = transform.localRotation;

            _dynamics = Dynamics.Mapper.GetDynamics(dynamicsType);
            _observer = Observers.Mapper.GetObserver(observerType);
            _rewarder = Rewards.Mapper.GetRewarder(rewarderType);

            GetComponent<BehaviorParameters>().BrainParameters.VectorObservationSize = _observer.Size;
        

            _material = GetComponent<Renderer>().material;
            _originalColor = _material.color;
            _bufferSensor = GetComponent<BufferSensorComponent>();

        }

        public override void OnEpisodeBegin()
        {
            base.OnEpisodeBegin();
            CollectedGoal = false;

        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);
            // Debug.Log($"{name} OnAction at step {GetComponentInParent<Statistician>().Time}");
            _dynamics.ProcessActions(actions, Rigidbody, moveSpeed, rotationSpeed, dragFactor, 3f);
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            // base.Heuristic(in actionsOut);

            var cActionsOut = actionsOut.ContinuousActions;

            var xValue = 0f;
            var zValue = 0f;
            Vector3 force;

            // Only for polar WASD controls
            // Ratio allows the agent to turn more or less in place, but still turn normally while moving.
            // The higher the ratio, the smaller circle the agent makes while turning in place (A/D)
            const float ratio = 1f;
        
            if (Input.GetKey(KeyCode.W)) xValue = 1f;
            if (Input.GetKey(KeyCode.S)) xValue = -1f;
        
            if (Input.GetKey(KeyCode.D)) zValue = 1f/ratio;
            if (Input.GetKey(KeyCode.A)) zValue = -1f/ratio;

            if (true)
            {
                force = new Vector3(xValue, 0, zValue);
            }

            cActionsOut[0] = force.x;
            cActionsOut[1] = force.z;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            base.CollectObservations(sensor);

            _observer.Observe(sensor, transform);
        
            var reward = _rewarder.ComputeReward(transform);
            
            AddReward(reward);
        
        
            // Collect Buffer observations
            var layerMask = 1 << 3; // Only look at the Agent layer
            var nearbyObjects =
                Physics.OverlapSphere(transform.position, Params.SightRadius, layerMask)
                    .Where(c => c.CompareTag("Agent") & c.transform != transform) // Get only agents 
                    .OrderBy(c => Vector3.Distance(c.transform.localPosition, transform.localPosition))
                    .Select(c => MLUtils.GetColliderInfo(transform, c))
                    .Take(Params.SightAgents);
        
            // Debug.Log(nearbyObjects);
            foreach (var agentInfo in nearbyObjects)
            {
                // Debug.Log(String.Join(",", agentInfo));
                _bufferSensor.AppendObservation(agentInfo);
            }

            // Draw some debugging lines
        
            Debug.DrawLine(transform.position, goal.position, Color.red, Time.fixedDeltaTime);
        
            // Debug.Log($"Current position: {transform.position}. Previous position: {PreviousPosition}");
        
            var parentPosition = transform.parent.position;
            var absPrevPosition = PreviousPosition + parentPosition;
            Debug.DrawLine(transform.position, absPrevPosition, Color.green, 20*Time.fixedDeltaTime);


        
            // Final updates
            PreviousPosition = transform.localPosition;
            _material.color = _originalColor;

        }


        private void OnTriggerStay(Collider other)
        {
            // Debug.Log("Hitting a trigger");
        
            if (other.name != goal.name) return;
        
            AddReward(_rewarder.TriggerReward(transform, other, true));
        
        
            CollectedGoal = true;

            GetComponentInParent<Manager>().ReachGoal(this);
            // Debug.Log("Trying to change color");
            _material.color = Color.blue;
            
            // Debug.Log("Collecting a reward");
        }

        protected void OnCollisionEnter(Collision other)
        {
            if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
            {
                Collision = 1;
                _material.color = Color.red;
            }
        
            AddReward(_rewarder.CollisionReward(transform, other, false));
        }
    
    
        protected void OnCollisionStay(Collision other)
        {
            if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
            {
                Collision = 1;
                _material.color = Color.red;
            }
        
            AddReward(_rewarder.CollisionReward(transform, other, true));
        }

        public Vector3 PreviousPosition { get; set; }
    }
}
