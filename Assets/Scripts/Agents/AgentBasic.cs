using System;
using System.Collections.Generic;
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
using UnityEngine.Serialization;
using System.Reflection;
using JetBrains.Annotations;
using Sensors;
// using RayPerceptionSensorComponent3D = Sensors.RaySensorComponent3D;

// Proposed reward structure:
// 16.5 total reward for approaching the goal
// 0.1 reward per decision step for reaching the goal (10 reward per 100 steps, max ~40)
// -0.01 reward per decision step for collisions (-1 reward per 100 steps)
// -0.01 reward per decision step

namespace Agents
{
    public class AgentBasic : Agent, IAgent
    {

        public int AgentIndex { get; set; }
        private Animator _animator;

        private int _animIDSpeed;
        
        private Material _material;
        private Color _originalColor;

        private BufferSensorComponent _bufferSensor;

        // Interface elements for all types of agents
        public Rigidbody Rigidbody { get; private set; }
        public Collider Collider { get; private set; }
        
        [field: SerializeField] public Transform Goal { get; set; }
        
        
        public Vector3 GoalScale { get; private set; }

        public bool controllable = true;

        // public bool velocityControl = false;
        public float maxSpeed = 2f;
        public float maxAcceleration = 5f;
        public float rotationSpeed = 3f;
        public float e_s = 2.23f;
        public float e_w = 1.26f;
        public float PreferredSpeed = 1.33f;

        // public float PreferredSpeed => Mathf.Sqrt(e_s / e_w);
        
        public float mass = 1f;

        // public DynamicsEnum dynamicsType;
        private IDynamics _dynamics;

        // public ObserversEnum observerType;
        private IObserver _observer;

        // public RewardersEnum rewarderType;
        // private IRewarder _rewarder;
        public IRewarder _rewarder { get; private set; }
        
        public Squasher.SquashersEnum squasherType;
        private Func<Vector2, Vector2> _squasher;


        // protected int Unfrozen = 1;

        internal int Collision = 0;




        private float _originalHeight;
        private float _originalGoalHeight;
        
        [Space(10)]
        [Header("Debug metrics")]
        [Space(10)]
        
        // Metrics
        public bool CollectedGoal;
        public bool rewardDisabled;
        public float distanceTraversed;
        public float energySpent;
        public float energySpentComplex;
        public float totalDistance;

        public bool afterFirstStep;

        public Dictionary<string, float> rewardParts;

        private bool _observeAcceleration = false;

        // Debug variables

        public float collisionReward;
        public float goalReward;
        public float potentialReward;
        public float speedReward;
        public float timeReward;
        public float standstillReward;
        public float energyReward;
        public float finalReward;
        public float totalReward;
        public Vector3 velocityDbg;
        public float speed;

        public bool debug;
        
        
        [CanBeNull] private RaySensorComponent _rayPerceptionSensor;

        public Vector3 PreviousPosition { get; set; }
        public Vector3 PreviousVelocity { get; set; }
        
        public Vector3 PreviousPositionPhysics { get; set; }
        public Vector3 PreviouserPositionPhysics { get; set; }
        // public Vector3 PreviousVelocityPhysics { get; set; }

        public List<Transform> neighborsOrder;

        private void Awake()
        {
            _rayPerceptionSensor = GetComponent<RaySensorComponent>();
            
            if (Params.DestroyRaycasts)
            {
                // Debug.Log("Destroying");
                DestroyImmediate(_rayPerceptionSensor);
                _rayPerceptionSensor = null;
            }

            if (_rayPerceptionSensor != null)
                _rayPerceptionSensor.RaysPerDirection = Params.RaysPerDirection;
            

            _observeAcceleration = Params.SightAcceleration;
            _bufferSensor = GetComponent<BufferSensorComponent>();
            
            if (_observeAcceleration)
                _bufferSensor.ObservableSize = 7;
            else 
                _bufferSensor.ObservableSize = 5;


        }

        public override void Initialize()
        {
            base.Initialize();
        
            Rigidbody = GetComponent<Rigidbody>();
            Collider = GetComponent<Collider>();
            _animator = GetComponentInChildren<Animator>();
            if (_animator != null)
            {
                _animIDSpeed = Animator.StringToHash("Speed");
            }
            _material = GetComponentInChildren<Renderer>().material;
            _originalColor = _material.color;
            _originalHeight = transform.localPosition.y;
            _originalGoalHeight = Goal.localPosition.y;
            PreviousVelocity = Vector3.zero;
            afterFirstStep = false;

            UpdateParams();

            GetComponent<BehaviorParameters>().BrainParameters.VectorObservationSize = _observer.Size;
        

            _bufferSensor.MaxNumObservables = Params.SightAgents;

            GoalScale = Goal.localScale;


            // Debug.Log($"Ray perception sensor: {_rayPerceptionSensor}");
            // Destroy(_rayPerceptionSensor);
            // Destroy(_bufferSensor);
            // _rayPerceptionSensor = null;

        }

        public override void OnEpisodeBegin()
        {
            base.OnEpisodeBegin();
            // Debug.Log("Starting episode");
            TeleportBack();
            PreviousPosition = transform.localPosition;
            PreviousVelocity = Vector3.zero;
            
            PreviousPositionPhysics = transform.localPosition;
            PreviouserPositionPhysics = transform.localPosition;
            // PreviousVelocityPhysics = Vector3.zero;

            afterFirstStep = false;

            CollectedGoal = false;
            rewardDisabled = false;
            energySpent = 0f;
            energySpentComplex = 0f;
            distanceTraversed = 0f;
            totalDistance = 0f;
            totalReward = 0f;
            rewardParts = new Dictionary<string, float>
            {
                ["collision"] = 0f,
                ["goal"] = 0f,
                ["potential"] = 0f,
                ["speed"] = 0f,
                ["time"] = 0f,
                ["standstill"] = 0f,
                ["energy"] = 0f,
                ["final"] = 0f,
            };
            
            UpdateParams();
            
            Rigidbody.constraints = RigidbodyConstraints.FreezeRotation | RigidbodyConstraints.FreezePositionY;
            Collider.enabled = true;

            _originalColor = _material.color;
        }

        public void FixedUpdate()
        {
            if (debug)
            {
                collisionReward = rewardParts["collision"];
                goalReward = rewardParts["goal"];
                potentialReward = rewardParts["potential"];
                speedReward = rewardParts["speed"];
                timeReward = rewardParts["time"];
                standstillReward = rewardParts["standstill"];
                energyReward = rewardParts["energy"];
                finalReward = rewardParts["final"];
                velocityDbg = Rigidbody.velocity;
                speed = Rigidbody.velocity.magnitude;
            }
        }

        public void Update()
        {
            _animator?.SetFloat(_animIDSpeed, Rigidbody.velocity.magnitude);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);

            // Debug.Log($"{name} OnAction at step {GetComponentInParent<Statistician>().Time}");
            
            PreviouserPositionPhysics = PreviousPositionPhysics;
            PreviousPositionPhysics = transform.localPosition;
            
            // PreviousVelocityPhysics = Rigidbody.velocity;
            
            if (!CollectedGoal)
            {
                _dynamics.ProcessActions(actions, Rigidbody, maxSpeed, maxAcceleration, rotationSpeed, _squasher);
            }


            var reward = _rewarder.ActionReward(transform, actions);
            AddReward(reward);
            
            
            // Update measurements
            // if (!CollectedGoal)
            // {
            //     // var velocity = Rigidbody.velocity;
            //     // Debug.Log($"Delta time: {Time.fixedDeltaTime}");
            //     
            //     var velocity = (transform.localPosition - PreviousPositionPhysics) / Time.fixedDeltaTime;
            //     var previousVelocity = (PreviousPositionPhysics - PreviouserPositionPhysics) / Time.fixedDeltaTime;
            //
            //     var (simpleEnergy, complexEnergy) =
            //         MLUtils.EnergyUsage(velocity, previousVelocity, e_s, e_w, Time.fixedDeltaTime);
            //     
            //     energySpent += simpleEnergy;
            //     energySpentComplex += complexEnergy;
            //     // TODO: Add non-finishing energy penalty
            //
            // }

            // Debug.Log($"Velocity from {PreviousVelocityPhysics} to {Rigidbody.velocity}");


            // Debug.Log(Rigidbody.velocity.magnitude);
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            // base.Heuristic(in actionsOut);

            if (!controllable) return;
            
            var cActionsOut = actionsOut.ContinuousActions;

            var xValue = 0f;
            var zValue = 0f;
            Vector3 force;

            // Only for polar WASD controls
            // Ratio allows the agent to turn more or less in place, but still turn normally while moving.
            // The higher the ratio, the smaller circle the agent makes while turning in place (A/D)
            // const float ratio = 1f;

            const float baseSpeed = 1.7f;
            
            if (Input.GetKey(KeyCode.W)) zValue = 1f;
            if (Input.GetKey(KeyCode.S)) zValue = -1f;

            if (Input.GetKey(KeyCode.D)) xValue = 1f;
            if (Input.GetKey(KeyCode.A)) xValue = -1f;
            
            // TODO: diagonals are faster with cartesian controls
            force = new Vector3(xValue, 0, zValue) * baseSpeed;

            if (Input.GetKey(KeyCode.LeftShift))
            {
                force *= 2;
            }
            else
            {
                force *= 0.5f;
            }
            
            // if (Input.GetKey(KeyCode.Space))
            // {
            //     Debug.Log("Hippity Hoppity");
            //     // Debug.Log(transform.localRotation);
            //     hopped = !hopped;
            //     if (hopped)
            //     {
            //         TeleportAway();
            //     }
            //     else
            //     {
            //         TeleportBack();
            //     }
            // }
            //
            
            // Debug.Log(force.magnitude);
            
            cActionsOut[0] = force.x;
            cActionsOut[1] = force.z;
        }

        public override void CollectObservations(VectorSensor sensor)
        {

            if (afterFirstStep) // Only compute reward after the first observation
            {
                var reward = _rewarder.ComputeReward(transform);
                AddReward(reward);
            }


            _observer.Observe(sensor, transform);

            var neighbors = _observer.ObserveAgents(_bufferSensor, transform, _observeAcceleration);
            neighborsOrder = neighbors;
            

            // if (IsMainAgent)
            // {
            //     Debug.Log($"Observing {neighbors.Count()} agents: {string.Join(", ", neighbors.Select(n => n.name))}");
            // }
            // Debug.Log($"Previous velocity: {PreviousVelocity}");
            // Debug.Log($"Current velocity: {Rigidbody.velocity}");
            

            // Draw some debugging lines
        
            // Debug.DrawLine(transform.position, goal.position, Color.red, Time.deltaTime*2);
        
            // Debug.Log($"Current position: {transform.position}. Previous position: {PreviousPosition}");
        
            // var parentPosition = transform.parent.position;
            // var absPrevPosition = PreviousPosition + parentPosition;
            // Debug.DrawLine(transform.position, absPrevPosition, Color.green, 20*Time.fixedDeltaTime);

            if (!CollectedGoal)
            {
                distanceTraversed += Vector3.Distance(transform.position, PreviousPosition);
                // Debug.Log($"Distance updated by {Vector3.Distance(transform.position, PreviousPosition)} to {distanceTraversed}");

            }
            totalDistance += Vector3.Distance(transform.position, PreviousPosition);
        
            // Final updates
            PreviousPosition = transform.localPosition;
            PreviousVelocity = Rigidbody.velocity;

            if (CollectedGoal) rewardDisabled = true;
            afterFirstStep = true;

            Collision = 0;
            // _material.color = _originalColor;

        }


        private void OnTriggerStay(Collider other)
        {
            // Debug.Log("Hitting a trigger");
        
            if (other.name != Goal.name) return;

            var currentSpeed = Rigidbody.velocity.magnitude;
            
            if (Params.GoalSpeedThreshold <= 0f || currentSpeed < Params.GoalSpeedThreshold) {
                AddReward(_rewarder.TriggerReward(transform, other, true));
                CollectGoal();
            }
            // Debug.Log("Trying to change color");
            // _material.color = Color.blue;
            
            // Debug.Log("Collecting a reward");
        }

        protected void OnCollisionEnter(Collision other)
        {
            if (other.collider.CompareTag("Ground")) return;
            if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
            {
                // Debug.Log(other.impulse);
                Collision = 1;
                // _material.color = Color.red;
            }
            // Debug.Log("Collision");
        
            AddReward(_rewarder.CollisionReward(transform, other, false));
        }
    
    
        protected void OnCollisionStay(Collision other)
        {
            if (other.collider.CompareTag("Ground")) return;
            if (other.collider.CompareTag("Obstacle") || other.collider.CompareTag("Agent"))
            {
                Collision = 1;
                // _material.color = Color.red;
            }
            // Debug.Log("Collision");
            
            // Debug.Log(other.impulse.magnitude / Time.fixedDeltaTime);
        
            AddReward(_rewarder.CollisionReward(transform, other, true));
        }


        public void SetColor(Color color, bool colorGoal = false)
        {
            _material.color = color;
            if (colorGoal)
            {
                Goal.GetComponent<Renderer>().material.color = color;
            }
        }

        private void UpdateParams()
        {
            _dynamics = Dynamics.Mapper.GetDynamics(Params.Dynamics);
            _observer = Observers.Mapper.GetObserver(Params.Observer);
            _rewarder = Rewards.Mapper.GetRewarder(Params.Rewarder);
            _squasher = Squasher.GetSquasher(squasherType);
            
            GetComponent<BehaviorParameters>().BrainParameters.VectorObservationSize = _observer.Size;

            if (_rayPerceptionSensor != null)
            {

                _rayPerceptionSensor.RayLayerMask = (1 << LayerMask.NameToLayer("Obstacle"));

                if (Params.RayAgentVision)
                {
                    _rayPerceptionSensor.RayLayerMask |= (1 << LayerMask.NameToLayer("Agent"));

                    _rayPerceptionSensor.RayLength = Params.RayLength;
                    _rayPerceptionSensor.MaxRayDegrees = Params.RayDegrees;

                }
            }
        }

        private void DLog(object message)
        {
            if (debug)
            {
                Debug.Log(message);
            }
        }

        private float GetCurrentReward()
        {
            var reward = (float) typeof(Agent).GetField("m_Reward", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(this);
            // Debug.Log(reward);
            return reward;
        }

        public new void AddReward(float reward)
        {
            base.AddReward(reward);
            totalReward += reward;
        }
        
        public void AddRewardPart(float reward, string type)
        {
            rewardParts[type] += reward;
        }
        
        public void AddReward(float reward, string type)
        {
            AddReward(reward);
            AddRewardPart(reward, type);
        }

        public void CollectGoal()
        {
            CollectedGoal = true;
            Manager.Instance.ReachGoal(this);
            Rigidbody.constraints = RigidbodyConstraints.FreezeAll;
            Collider.enabled = false;

            if (!Params.EvaluationMode)
            {
                TeleportAway();
            }
        }
        public void TeleportAway()
        {
            // Debug.Log("Teleporting away");
            var newPosition = transform.localPosition;
            newPosition.y = -10f;
            transform.localPosition = newPosition;
            
            transform.localRotation *= Quaternion.Euler(90f, 0f, 0f);
            
            var newGoalPosition = Goal.transform.localPosition;
            newGoalPosition.y = -10f;
            Goal.transform.localPosition = newGoalPosition;
            
            // Debug.Log("New position: " + transform.localPosition);
        }
        
        public void TeleportBack()
        {
            // Debug.Log("Teleporting back");
            var newPosition = transform.localPosition;
            newPosition.y = _originalHeight;
            transform.localPosition = newPosition;
            
            var newRotation = transform.localRotation;
            newRotation.x = 0f;
            newRotation.z = 0f;
            transform.localRotation = newRotation;

            
            var newGoalPosition = Goal.transform.localPosition;
            newGoalPosition.y = _originalGoalHeight;
            Goal.transform.localPosition = newGoalPosition;

            // Debug.Log("New position: " + transform.localPosition);
        }
        
        private bool IsMainAgent => name == "Person";
        
        public void AddFinalReward()
        {
            if (CollectedGoal)
            {
                AddReward(_rewarder.FinishReward(transform, true));
            }
            else
            {
                AddReward(_rewarder.FinishReward(transform, false));
            }
        }


        public void RecordEnergy()
        {
            if (!CollectedGoal)
            {
                // var velocity = Rigidbody.velocity;
                // Debug.Log($"Delta time: {Time.fixedDeltaTime}");
                
                var velocity = (transform.localPosition - PreviousPositionPhysics) / Time.fixedDeltaTime;
                var previousVelocity = (PreviousPositionPhysics - PreviouserPositionPhysics) / Time.fixedDeltaTime;

                var (simpleEnergy, complexEnergy) =
                    MLUtils.EnergyUsage(velocity, previousVelocity, e_s, e_w, Time.fixedDeltaTime);
                
                energySpent += simpleEnergy;
                energySpentComplex += complexEnergy;
                // TODO: Add non-finishing energy penalty

            }
        }


    }
}
