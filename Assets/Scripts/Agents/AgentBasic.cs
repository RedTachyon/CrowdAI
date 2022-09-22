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
    public class AgentBasic : Agent
    {
    
        private Material _material;
        private Color _originalColor;

        private BufferSensorComponent _bufferSensor;
    
        [HideInInspector] public Rigidbody Rigidbody;
        [HideInInspector] public Collider Collider;

        public bool controllable = true;

        // public bool velocityControl = false;
        public float maxSpeed = 2f;
        public float maxAcceleration = 5f;
        public float rotationSpeed = 3f;

        public float mass = 1f;

        // public DynamicsEnum dynamicsType;
        private IDynamics _dynamics;

        // public ObserversEnum observerType;
        private IObserver _observer;

        public RewardersEnum rewarderType;
        private IRewarder _rewarder;

        public Squasher.SquashersEnum squasherType;
        private Func<Vector2, Vector2> _squasher;


        // protected int Unfrozen = 1;

        internal int Collision = 0;


        public Transform goal;
        public Vector3 goalScale;

        private float _originalHeight;
        private float _originalGoalHeight;
        
        [Space(10)]
        [Header("Debug metrics")]
        [Space(10)]
        
        // Metrics
        public bool CollectedGoal;
        public float distanceTraversed;
        public float energySpent;
        public float totalDistance;

        public Dictionary<string, float> rewardParts;

        private bool _observeAcceleration = false;

        // Debug variables

        public float collisionReward;
        public float goalReward;
        public float potentialReward;
        public float speedReward;
        public float timeReward;
        public float standstillReward;
        public float totalReward;
        public Vector3 velocity;
        public float speed;

        public bool debug;
        
        
        [CanBeNull] private RaySensorComponent _rayPerceptionSensor;

        public Vector3 PreviousPosition { get; set; }
        public Vector3 PreviousVelocity { get; set; }

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
            _material = GetComponent<Renderer>().material;
            _originalColor = _material.color;
            _originalHeight = transform.localPosition.y;
            _originalGoalHeight = goal.localPosition.y;
            PreviousVelocity = Vector3.zero;

            UpdateParams();

            GetComponent<BehaviorParameters>().BrainParameters.VectorObservationSize = _observer.Size;
        

            _bufferSensor.MaxNumObservables = Params.SightAgents;

            goalScale = goal.localScale;


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
            PreviousVelocity = Vector3.zero;

            CollectedGoal = false;
            energySpent = 0f;
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
                velocity = Rigidbody.velocity;
                speed = Rigidbody.velocity.magnitude;
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);

            // Debug.Log($"{name} OnAction at step {GetComponentInParent<Statistician>().Time}");
            
            if (!CollectedGoal)
            {
                _dynamics.ProcessActions(actions, Rigidbody, maxSpeed, maxAcceleration, rotationSpeed, _squasher);
            }


            var reward = _rewarder.ActionReward(transform, actions);
            AddReward(reward);
            
            
            // Update measurements
            if (!CollectedGoal)
            {
                energySpent += Params.E_s * Time.fixedDeltaTime;
                energySpent += Params.E_w * Rigidbody.velocity.sqrMagnitude * Time.fixedDeltaTime;
                // energySpent += Params.E_a * Rigidbody.angularVelocity.sqrMagnitude * Time.fixedDeltaTime;
                // TODO: rotational energy?
            }



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
            

            force = new Vector3(xValue, 0, zValue).normalized * baseSpeed;

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

            // Debug.Log($"Collected goal? {CollectedGoal}");

            var reward = _rewarder.ComputeReward(transform);
            AddReward(reward);
            

            // DLog($"Current reward: {GetCurrentReward()}");
            
            _observer.Observe(sensor, transform);

            _observer.ObserveAgents(_bufferSensor, transform, _observeAcceleration);
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
            }
            totalDistance += Vector3.Distance(transform.position, PreviousPosition);
        
            // Final updates
            PreviousPosition = transform.localPosition;
            PreviousVelocity = Rigidbody.velocity;

            Collision = 0;
            // _material.color = _originalColor;

        }


        private void OnTriggerStay(Collider other)
        {
            // Debug.Log("Hitting a trigger");
        
            if (other.name != goal.name) return;

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
                Debug.Log(other.impulse);
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

        private new void AddReward(float reward)
        {
            base.AddReward(reward);
            totalReward += reward;
        }
        
        public void AddRewardPart(float reward, string type)
        {
            rewardParts[type] += reward;
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
            
            var newGoalPosition = goal.transform.localPosition;
            newGoalPosition.y = -10f;
            goal.transform.localPosition = newGoalPosition;
            
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

            
            var newGoalPosition = goal.transform.localPosition;
            newGoalPosition.y = _originalGoalHeight;
            goal.transform.localPosition = newGoalPosition;

            // Debug.Log("New position: " + transform.localPosition);
        }
        
    }
}
