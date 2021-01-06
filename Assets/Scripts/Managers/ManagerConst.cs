using System;
using Unity.MLAgents;
using UnityEngine;

public class ManagerConst : Statistician
{

    private Transform _agent;

    public override void Initialize()
    {
        base.Initialize();
    
        _agent = GetComponentInChildren<AgentConst>().transform;

    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        // Debug.Log("Manager starting an episode");
        // _done = false;
        // Debug.Log(UnityEngine.Random.state.GetHashCode());
        // UnityEngine.Random.InitState(DateTime.Now.Millisecond);

        var agent = _agent.GetComponent<AgentConst>();
        _agent.localPosition = agent.startPosition;
        _agent.localRotation = agent.startRotation;
        
        agent.GetComponent<Rigidbody>().velocity = Vector3.zero;
        agent.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;

        agent.GetComponent<AgentConst>().PreviousPosition = _agent.localPosition;

    }

    public override void Heuristic(float[] actionsOut)
    {
        if (Input.GetKey("space"))
        {
            EndEpisode();
        }
        actionsOut[0] = 0f;
    }

    public new void ReachGoal(Walker agent)
    {
        base.ReachGoal(agent);
        // Debug.Log("I'm here!");
        // agent.goal.localPosition = new Vector3(
        //     UnityEngine.Random.Range(-9f, 9f),
        //     0.15f,
        //     UnityEngine.Random.Range(-9f, 9f));
    }
}
