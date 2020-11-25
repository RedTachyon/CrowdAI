using Unity.MLAgents.Sensors;
using UnityEngine;


public class AgentSensorComponent : SensorComponent
{
    public string sensorName;
    public float radius;
    
    public override ISensor CreateSensor()
    {
        var sensor = new AgentSensor(sensorName, radius);
        return sensor;
    }

    public override int[] GetObservationShape()
    {
        throw new System.NotImplementedException();
    }
}
