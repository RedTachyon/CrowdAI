using UnityEngine;

namespace Agents
{
    public interface IAgent
    {
        public Rigidbody Rigidbody { get; }
        public Collider Collider { get;  }
    }
}