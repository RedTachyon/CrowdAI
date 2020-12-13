using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.MLAgents.SideChannels;

public class Manager : Statistician
{
    private Dictionary<Transform, Vector3> _startPos;
    private Dictionary<Transform, Quaternion> _startRot;

    public override void Initialize()
    {
        

        QualitySettings.vSyncCount = 0;
        _finished = new Dictionary<Transform, bool>();
        _startPos = new Dictionary<Transform, Vector3>();
        _startRot = new Dictionary<Transform, Quaternion>();
        
        foreach (Transform agent in transform)
        {
            // Get each agent's starting position
            _startPos[agent] = agent.position;
            _startRot[agent] = agent.rotation;
        }
        
    }
    
    

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
                
                agent.position = _startPos[agent];
                agent.rotation = _startRot[agent];
        
                agent.GetComponent<Rigidbody>().velocity = Vector3.zero;
                agent.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        
                agent.GetComponent<AgentController>().PreviousPosition = _startPos[agent];
            }
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        actionsOut[0] = 0f;
    }
}
