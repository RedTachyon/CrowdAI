using Unity.MLAgents.Sensors;
using UnityEngine;


public class AgentSensorComponent : SensorComponent
{
    public string sensorName;
    public float radius;
    public int numAgents = 5;
    public int maxObjects = 20;
    
    public override ISensor CreateSensor()
    {
        var sensor = new AgentSensor(sensorName, radius, transform, maxObjects, numAgents);
        return sensor;
    }

    public override int[] GetObservationShape()
    {
        int[] shape = {numAgents * 4}; // p.x, p.z, v.x., v.z
        return shape;
    }
}
