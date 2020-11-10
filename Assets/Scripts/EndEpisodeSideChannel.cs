using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using System.Text;
using System;

public class EndEpisodeSideChannel : SideChannel
{
    public EndEpisodeSideChannel()
    {
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        var receivedString = msg.ReadString();
        Debug.Log("From Python : " + receivedString);
    }

    public void SendEpisodeStatus(bool status)
    {
        // Debug.Log("Sending a message!");
        using (var msgOut = new OutgoingMessage())
        {
            msgOut.WriteBoolean(status);
            QueueMessageToSend(msgOut);
        }
    }
    
}