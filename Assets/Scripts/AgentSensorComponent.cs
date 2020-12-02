using Unity.MLAgents.Sensors;
using UnityEngine;


public class AgentSensorComponent : SensorComponent
{
    public string sensorName;
    public float radius;
    public int maxAgents;
    
    public override ISensor CreateSensor()
    {
        var sensor = new AgentSensor(sensorName, radius, transform, maxAgents);
        return sensor;
    }

    public override int[] GetObservationShape()
    {
        int[] shape = {3};
        return shape;
    }
}
