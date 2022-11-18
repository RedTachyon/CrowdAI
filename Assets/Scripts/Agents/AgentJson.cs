using System;
using UnityEngine;

namespace Agents
{
    public class AgentJson : MonoBehaviour, IAgent
    {
        public Rigidbody Rigidbody { get; private set; }
        public Collider Collider { get; private set; }
        
        private void Awake()
        {
            Rigidbody = GetComponent<Rigidbody>();
            Collider = GetComponent<Collider>();
        }
    }
}