using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.MLAgents.SideChannels;

public class ManagerRandom : Statistician
{

    
    public override void OnEpisodeBegin()
    {
        // Debug.Log("Manager starting an episode");
        _finished.Clear();
        // _done = false;
        
        foreach (Transform agent in transform)
        {
            if (agent.gameObject.activeSelf)
            {
                _finished[agent] = false;
                // agent.GetComponent<Controller>().Unfreeze();
                
                agent.localPosition = new Vector3(
                    UnityEngine.Random.Range(-9f, 9f), 
                    0.25f,
                    UnityEngine.Random.Range(-9f, 9f)
                );

                agent.GetComponent<AgentRandom>().goal.localPosition = new Vector3(
                    UnityEngine.Random.Range(-9f, 9f),
                    0.15f,
                    UnityEngine.Random.Range(-9f, 9f)
                );
                
                agent.localRotation = Quaternion.Euler(0f, UnityEngine.Random.Range(0f, 360f), 0f);
        
                agent.GetComponent<Rigidbody>().velocity = Vector3.zero;
                agent.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        
                agent.GetComponent<AgentRandom>().PreviousPosition = agent.localPosition;
            }
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        actionsOut[0] = 0f;
    }

    public new void ReachGoal(Walker agent)
    {
        base.ReachGoal(agent);
        Debug.Log("I'm here!");
        agent.goal.localPosition = new Vector3(
            UnityEngine.Random.Range(-9f, 9f),
            0.15f,
            UnityEngine.Random.Range(-9f, 9f));
    }
}
