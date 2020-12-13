using System;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.UIElements;

public class AgentSensor : ISensor
{
    float _radius;
    string _name;
    int _numAgents; // How many agents (at most) do you want to observe
    int _maxObjects; // How many objects can be perceived (for Physics.OverlapSphereNonAlloc)

    public Transform Transform;

    public AgentSensor(string name, float radius, Transform transform, int maxObjects, int numAgents)
    {
        _name = name;
        _radius = radius;
        Transform = transform;
        _maxObjects = maxObjects; // not used yet
        _numAgents = numAgents;

    }
    
    // IDEA: use Physics.OverlapSphere to find all objects in range -> sort them by distance? -> get the ones in front of object via some smart dot product
    public int[] GetObservationShape()
    {
        int[] shape = {_numAgents * 4};
        return shape;
    }

    public int Write(ObservationWriter writer)
    {
        const float nan = 1e6f;
        
        var colliders = Physics.OverlapSphere(Transform.localPosition, _radius)
            .Where(c => c.CompareTag("Agent"))
            .Where(c => c.transform != Transform)
            .OrderBy(c => Vector3.Distance(Transform.localPosition, c.transform.localPosition))
            .Take(_numAgents)
            .ToArray();
        
        var debugOut = new List<float>();

        var offset = 0;
        foreach (var collider in colliders)
        {
            Debug.DrawLine(Transform.position, collider.transform.position, Color.red, 0.1f);
            var position = collider.transform.localPosition;
            var velocity = collider.attachedRigidbody.velocity;
            var vals = new[] {position.x, position.z, velocity.x, velocity.z};
            writer.AddRange(vals, offset);
            offset += 4;

            debugOut.AddRange(vals);
        }

        var numValues = offset;

        for (var i = colliders.Length; i < _numAgents; i++)
        {
            var vals = new[] {nan, nan, nan, nan};
            // var vals = new[] {0f, 0f, 0f, 0f};
            writer.AddRange(vals, offset);
            offset += 4;
            
            debugOut.AddRange(vals);
        }
        
        Debug.Log("Agents sensed = [" + string.Join(",",
            debugOut
                .ConvertAll(i => i.ToString())
                .ToArray()) + "]");

        return numValues;
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
