using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using System.Text;
using System;

public class StatsSideChannel : SideChannel
{
    public StatsSideChannel()
    {
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        var receivedString = msg.ReadString();
        Debug.Log("From Python : " + receivedString);
    }

    public void SendMessage(string msg)
    {
        var stringToSend = msg;
        using (var msgOut = new OutgoingMessage())
        {
            msgOut.WriteString(stringToSend);
            QueueMessageToSend(msgOut);
        }
    }

    // public void SendDebugStatementToPython(string logString, string stackTrace, LogType type)
    // {
    //     if (type == LogType.Error)
    //     {
    //         var stringToSend = type.ToString() + ": " + logString + "\n" + stackTrace;
    //         using (var msgOut = new OutgoingMessage())
    //         {
    //             msgOut.WriteString(stringToSend);
    //             QueueMessageToSend(msgOut);
    //         }
    //     }
    // }
}