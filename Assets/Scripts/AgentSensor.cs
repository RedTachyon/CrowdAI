using System.Linq;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class AgentSensor : ISensor
{
    float _radius;
    string _name;
    int _numAgents;

    public AgentSensor(string name, float radius)
    {
        _name = name;
        _radius = radius;

    }
    
    // IDEA: use Physics.OverlapSphere to find all objects in range -> sort them by distance? -> get the ones in front of object via some smart dot product
    public int[] GetObservationShape()
    {
        int[] shape = {3};
        return shape;
    }

    public int Write(ObservationWriter writer)
    {
        var colliders = Physics.OverlapSphere(Vector3.zero, _radius)
            .Where(collider => collider.CompareTag("Agent"))
            as Collider[];
        
        float[] obs;

        if (colliders == null || colliders.Length == 0)
        {
            obs = new[] {0f, 0f, 0f};
        }
        else
        {
            var pos = colliders[0].transform.localPosition;
            obs = new[] {pos.x, pos.y, pos.z};
        }
        
        writer.AddRange(obs);

        return 3;
    }

    public byte[] GetCompressedObservation()
    {
        return null;
    }

    public void Update() {}

    public void Reset() {}

    public SensorCompressionType GetCompressionType()
    {
        return SensorCompressionType.None;
    }

    public string GetName()
    {
        return _name;
    }
}
