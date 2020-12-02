using System.Linq;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.UIElements;

public class AgentSensor : ISensor
{
    float _radius;
    string _name;
    int _numAgents;
    int _maxAgents;

    public Transform Transform;

    public AgentSensor(string name, float radius, Transform transform, int maxAgents)
    {
        _name = name;
        _radius = radius;
        Transform = transform;
        _maxAgents = maxAgents;

    }
    
    // IDEA: use Physics.OverlapSphere to find all objects in range -> sort them by distance? -> get the ones in front of object via some smart dot product
    public int[] GetObservationShape()
    {
        int[] shape = {3};
        return shape;
    }

    public int Write(ObservationWriter writer)
    {
        
        var colliders = Physics.OverlapSphere(Transform.localPosition, _radius)
            .Where(collider => collider.CompareTag("Agent"))
            .Where(collider => collider.transform != Transform)
            .ToArray();
        
        float[] obs;
        
        // Debug.Log(colliders.Length);

        foreach (var collider in colliders)
        {
            Debug.DrawLine(Transform.position, collider.transform.position, Color.red);
        }


        if (colliders.Length == 0)
        {
            obs = new[] {0f, 0f, 0f};
        }
        else
        {
            var pos = colliders[0].transform.localPosition;
            // Debug.Log("Something's actually detected");
            obs = new[] {pos.x, pos.y, pos.z};
        }
        
        writer.AddRange(obs);
        
        // Debug.Log(obs[0]);


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
