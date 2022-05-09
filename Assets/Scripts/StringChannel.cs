using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using System.Text;
using System;
using System.Collections.Generic;
using Managers;

public class StringChannel : SideChannel
{
    private Dictionary<string, string> _values;
    public StringChannel()
    {
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f8");
        _values = new Dictionary<string, string>();
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        var key = msg.ReadString();
        var value = msg.ReadString();
        _values[key] = value;
        // Debug.Log($"Received message: {key}: {value}");
        // Manager.Instance.savePath = receivedString;
    }

    public string GetWithDefault(string key, string defaultValue)
    {
        return _values.ContainsKey(key) ? _values[key] : defaultValue;
    }
}