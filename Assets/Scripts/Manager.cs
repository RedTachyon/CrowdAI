using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using UnityEngine;
using Unity.MLAgents.SideChannels;

public class Manager : Agent
{
    // TODO: change this into a MonoBehavior, and use the Academy's resetting behaviors
    private Dictionary<Transform, bool> _finished;
    private Dictionary<Transform, Vector3> _startPos;
    private Dictionary<Transform, Quaternion> _startRot;

    // private EndEpisodeSideChannel _endChannel;
    // private bool _done;
    
    public override void Initialize()
    {
        // _endChannel = new EndEpisodeSideChannel();
        
        // SideChannelsManager.RegisterSideChannel(_endChannel);

        QualitySettings.vSyncCount = 0;
        _finished = new Dictionary<Transform, bool>();
        _startPos = new Dictionary<Transform, Vector3>();
        _startRot = new Dictionary<Transform, Quaternion>();
        
        foreach (Transform agent in transform)
        {
            // Get each agent's starting position
            _startPos[agent] = agent.localPosition;
            _startRot[agent] = agent.localRotation;

        }
        
    }

    public void ReachGoal(Controller agent)
    {
        _finished[agent.transform] = true;
        // agent.EndEpisode();
        agent.Freeze();

        if (!_finished.Values.Contains(false))
        {
            
            // _done = true;
        }
    }


    public override void OnEpisodeBegin()
    {
        Debug.Log("Manager starting an episode");
        _finished.Clear();
        // _done = false;
        
        foreach (Transform agent in transform)
        {
            if (agent.gameObject.activeSelf)
            {
                _finished[agent] = false;
                agent.GetComponent<Controller>().Unfreeze();
                
                agent.localPosition = _startPos[agent];
                agent.localRotation = _startRot[agent];
        
                agent.GetComponent<Rigidbody>().velocity = Vector3.zero;
                agent.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        
                agent.GetComponent<Controller>().PreviousPosition = _startPos[agent];
            }
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        
    }
}
