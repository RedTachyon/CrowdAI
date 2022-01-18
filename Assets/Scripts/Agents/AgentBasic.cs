using System;
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
        protected Collider Collider;

        // public bool velocityControl = false;
        public float moveSpeed = 25f;
        public float accelerationMax = 5f;
        public float rotationSpeed = 3f;
        public float dragFactor = 5f;

        // public DynamicsEnum dynamicsType;
        private IDynamics _dynamics;

        // public ObserversEnum observerType;
        private IObserver _observer;

        public RewardersEnum rewarderType;
        private IRewarder _rewarder;

        public Squasher.SquashersEnum squasherType;
        private Func<Vector2, Vector2> _squasher;


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
            Collider = GetComponent<Collider>();
            // startY = transform.localPosition.y;
            startPosition = transform.localPosition;
            startRotation = transform.localRotation;

            UpdateParams();

            GetComponent<BehaviorParameters>().BrainParameters.VectorObservationSize = _observer.Size;
        

            _material = GetComponent<Renderer>().material;
            _originalColor = _material.color;
            _bufferSensor = GetComponent<BufferSensorComponent>();
            _bufferSensor.MaxNumObservables = Params.SightAgents;

        }

        public override void OnEpisodeBegin()
        {
            base.OnEpisodeBegin();
            CollectedGoal = false;
            
            UpdateParams();

            if (Params.EvaluationMode)
            {
                Rigidbody.constraints = RigidbodyConstraints.FreezeRotation | RigidbodyConstraints.FreezePositionY;
                // Collider.enabled = true;
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);
            // Debug.Log($"{name} OnAction at step {GetComponentInParent<Statistician>().Time}");
            
            if (!CollectedGoal || !Params.EvaluationMode)
            {
                _dynamics.ProcessActions(actions, Rigidbody, moveSpeed, moveSpeed, rotationSpeed, _squasher);
            }


            if (!CollectedGoal)
            {
                var reward = _rewarder.ActionReward(transform, actions);
                AddReward(reward);
            }
            // Debug.Log(Rigidbody.velocity.magnitude);
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
            // const float ratio = 1f;
        
            if (Input.GetKey(KeyCode.W)) zValue = 1f;
            if (Input.GetKey(KeyCode.S)) zValue = -1f;

            if (Input.GetKey(KeyCode.D)) xValue = 1f;
            if (Input.GetKey(KeyCode.A)) xValue = -1f;
            

            force = new Vector3(xValue, 0, zValue).normalized;

            if (Input.GetKey(KeyCode.LeftShift))
            {
                force *= 2;
            }
            else
            {
                force *= 0.5f;
            }
            
            // Debug.Log(force.magnitude);
            
            cActionsOut[0] = force.x;
            cActionsOut[1] = force.z;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            base.CollectObservations(sensor);

            if (!CollectedGoal)
            {
                var reward = _rewarder.ComputeReward(transform);
                AddReward(reward);
            }
            
            _observer.Observe(sensor, transform);

            _observer.ObserveAgents(_bufferSensor, transform);
            

            // Draw some debugging lines
        
            Debug.DrawLine(transform.position, goal.position, Color.red, Time.deltaTime*2);
        
            // Debug.Log($"Current position: {transform.position}. Previous position: {PreviousPosition}");
        
            var parentPosition = transform.parent.position;
            var absPrevPosition = PreviousPosition + parentPosition;
            Debug.DrawLine(transform.position, absPrevPosition, Color.green, 20*Time.fixedDeltaTime);


        
            // Final updates
            PreviousPosition = transform.localPosition;
            Collision = 0;
            // _material.color = _originalColor;

        }


        private void OnTriggerStay(Collider other)
        {
            // Debug.Log("Hitting a trigger");
        
            if (other.name != goal.name) return;
        
            AddReward(_rewarder.TriggerReward(transform, other, true));
        
        
            CollectedGoal = true;
            Manager.Instance.ReachGoal(this);

            if (Params.EvaluationMode)
            {
                Rigidbody.constraints = RigidbodyConstraints.FreezeAll;
                // Collider.enabled = false;
            }
            // Debug.Log("Trying to change color");
            // _material.color = Color.blue;
            
            // Debug.Log("Collecting a reward");
        }

        protected void OnCollisionEnter(Collision other)
        {
            if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
            {
                Collision = 1;
                // _material.color = Color.red;
            }
        
            AddReward(_rewarder.CollisionReward(transform, other, false));
        }
    
    
        protected void OnCollisionStay(Collision other)
        {
            if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
            {
                Collision = 1;
                // _material.color = Color.red;
            }
        
            AddReward(_rewarder.CollisionReward(transform, other, true));
        }

        public Vector3 PreviousPosition { get; set; }

        public void SetColor(Color color, bool colorGoal = false)
        {
            _material.color = color;
            if (colorGoal)
            {
                goal.GetComponent<Renderer>().material.color = color;
            }
        }

        private void UpdateParams()
        {
            _dynamics = Dynamics.Mapper.GetDynamics(Params.Dynamics);
            _observer = Observers.Mapper.GetObserver(Params.Observer);
            _rewarder = Rewards.Mapper.GetRewarder(rewarderType);
            _squasher = Squasher.GetSquasher(squasherType);
            
            GetComponent<BehaviorParameters>().BrainParameters.VectorObservationSize = _observer.Size;

        }
    }
}
