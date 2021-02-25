using UnityEngine;
using Unity.MLAgents.SideChannels;
using Unity.MLAgents;

public class StatsCommunicator : MonoBehaviour
{
    public StatsSideChannel StatsChannel;
    private void Awake()
    {
        StatsChannel = new StatsSideChannel();
        SideChannelManager.RegisterSideChannel(StatsChannel);
    }

    private void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            SideChannelManager.UnregisterSideChannel(StatsChannel);
        }
    }
}