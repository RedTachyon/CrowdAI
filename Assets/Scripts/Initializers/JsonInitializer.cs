using System;
using System.Collections.Generic;
using System.IO;
using Agents;
using Newtonsoft.Json;
using UnityEngine;

namespace Initializers
{

    public class TrajectoryData
    {
        public float[] time { get; set; }
        public float[,,] position { get; set; }

        public TrajectoryData(float[] _time, float[,,] _position)
        {
            time = _time;
            position = _position;
        }
    }
    public class JsonInitializer : IInitializer
    {
        private List<Vector3> _spawnPoints = new();
        private List<Vector3> _goalPoints = new();
        
        public JsonInitializer(string path)
        {
            // Debug.Log(Directory.GetCurrentDirectory());
            if (path == null)
            {
                throw new ArgumentNullException(nameof(path));
            }
            
            string fullPath = Path.Combine(Directory.GetCurrentDirectory(), "data", path);
            
            Debug.Log($"Trying to read from {fullPath}");
            var data = JsonConvert.DeserializeObject<TrajectoryData>(File.ReadAllText(fullPath));
            
            Debug.Log(data.position.GetLength(0));
            var numAgents = data.position.GetLength(0);
            var numTimesteps = data.position.GetLength(1);

            for (var i = 0; i < numAgents; i++)
            {
                var spawn = new Vector3(
                    data.position[i, 0, 0],
                    0f,
                    data.position[i, 0, 1]
                );
                var goal = new Vector3(
                    data.position[i, numTimesteps-1, 0],
                    0f,
                    data.position[i, numTimesteps-1, 1]
                );
                
                _spawnPoints.Add(spawn);
                _goalPoints.Add(goal);
            }

            // _spawnPoints.Add();

        }
        public void PlaceAgents(Transform baseTransform)
        {
            var placedAgents = new List<Vector3>();
            var placedGoals = new List<Vector3>();
            var agentIdx = 0;

            var numAgents = baseTransform.childCount;

            foreach (Transform agent in baseTransform)
            {  
                var goal = agent.GetComponent<AgentBasic>().goal;

                // var xSpawn = _spawnPoints[agentIdx];
                // var ySpawn
                var xSpawn = _spawnPoints[agentIdx].x;
                var zSpawn = _spawnPoints[agentIdx].z;
                
                var xGoal = _goalPoints[agentIdx].x;
                var zGoal = _goalPoints[agentIdx].z;
                
                var newPosition = new Vector3(xSpawn, agent.localPosition.y, zSpawn);
                var goalPosition = new Vector3(xGoal, goal.localPosition.y, zGoal);
                var newRotation = Quaternion.LookRotation(goalPosition, Vector3.up);

                agent.localPosition = newPosition;
                agent.localRotation = newRotation;
                goal.localPosition = goalPosition;
            
                // Save the placed agents
                placedAgents.Add(newPosition);
                placedGoals.Add(goalPosition);

                // Reset the dynamics
                agent.GetComponent<Rigidbody>().velocity = Vector3.zero;
                agent.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        
                agent.GetComponent<AgentBasic>().PreviousPosition = agent.localPosition;
                
                agentIdx++;
            }
            
        }
    }
}