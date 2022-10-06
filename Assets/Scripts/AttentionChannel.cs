using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using System.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using Managers;

public class AttentionChannel : SideChannel
{
    [CanBeNull] private string _value;
    public int[][] Attention { get; private set; } 
    
    public AttentionChannel()
    {
        ChannelId = new Guid("43842a14-43d6-11ed-8437-acde48001122");
        _value = SAMPLE_MESSAGE;
        Attention = ParseMessage(_value);
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        var value = msg.ReadString();
        _value = value;
        Attention = ParseMessage(value);
        // Debug.Log($"Received message: {key}: {value}");
        // Manager.Instance.savePath = receivedString;
    }

    protected static int[][] ParseMessage(string message)
    {
        var values = message.Split("\n").Select((line) => line.Split(" ").Select(Int32.Parse).ToArray()).ToArray();
        return values;
    }

    private string SAMPLE_MESSAGE =
        "10 13 4 4 21 21 3 10 3 10 0 0\n4 7 22 4 5 27 12 12 4 3 0 0\n6 3 14 20 3 19 3 19 2 12 0 0\n17 5 14 5 22 3 15 3 14 3 0 0\n5 9 21 8 7 19 5 19 6 0 0 0\n15 8 3 29 2 26 2 2 14 0 0 0\n3 25 6 25 4 14 2 3 14 3 0 0\n25 4 15 3 4 17 16 3 11 3 0 0\n3 18 4 26 25 3 4 14 3 0 0 0\n19 5 25 4 13 3 16 3 12 0 0 0\n17 3 13 4 24 4 17 15 4 0 0 0\n5 12 3 28 6 17 4 14 12 0 0 0\n14 4 4 28 2 18 13 3 11 2 0 0\n22 6 19 17 5 5 12 4 12 0 0 0\n18 5 22 2 15 4 14 4 12 4 0 0\n5 28 4 20 3 19 15 3 3 0 0 0\n4 20 4 20 4 15 3 14 14 3 0 0\n25 4 17 3 3 16 15 3 2 12 0 0\n18 3 27 3 23 3 17 3 3 0 0 0\n3 23 23 3 17 3 3 12 3 12 0 0\n10 3 26 4 18 4 19 4 12 0 0 0\n5 26 3 4 18 19 3 10 3 10 0 0\n4 22 4 29 3 11 11 3 10 3 0 0\n24 4 13 15 4 4 14 19 3 0 0 0\n6 10 14 4 15 4 24 4 4 15 0 0\n11 17 5 4 29 3 13 3 12 3 0 0\n3 19 31 5 4 16 3 16 3 0 0 0\n16 2 2 24 3 24 3 13 2 11 0 0\n5 19 3 4 27 7 17 4 14 0 0 0\n20 4 3 24 15 3 3 12 11 3 0 0\n4 25 17 7 14 3 13 3 11 4 0 0\n15 5 15 4 16 13 4 3 11 13 0 0\n5 20 4 20 15 3 11 15 3 3 0 0\n23 5 18 5 4 12 19 3 11 0 0 0\n13 4 4 30 14 4 12 4 11 3 0 0\n5 4 33 19 4 16 4 13 3 0 0 0\n15 4 20 5 15 4 15 4 14 4 0 0\n14 32 3 16 3 2 14 12 2 2 0 0\n14 4 6 21 16 4 4 14 3 13 0 0\n6 29 18 4 5 15 4 14 4 0 0 0\n16 5 17 5 16 4 19 5 13 0 0 0\n5 18 18 4 4 23 12 3 10 4 0 0\n20 5 5 23 16 4 12 4 12 0 0 0\n14 4 23 4 16 4 17 3 12 3 0 0\n7 9 6 17 21 5 5 16 14 0 0 0\n11 4 23 28 3 2 16 2 12 0 0 0\n11 4 5 30 4 16 3 12 3 12 0 0\n5 26 4 16 7 4 12 3 13 11 0 0\n13 5 17 4 23 13 4 4 15 3 0 0\n5 17 27 4 12 3 13 2 3 13 0 0";
}