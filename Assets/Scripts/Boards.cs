using System;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class Boards : MonoBehaviour
{
    [Range(1, 8)]
    public int boards = 1;
    public float distance = 20;
    private void Awake()
    {
        // var board = GetComponentInChildren<Transform>();
        var board = transform.GetChild(0);
        // board.localPosition = GetPosition(1);
        
        // Debug.Log(board.name);


        for (var i = 1; i < boards; i++)
        {
            var newBoard = Instantiate(board, transform);
            newBoard.name = board.name + $" ({i})";
            newBoard.localPosition = GetPosition(i);
        }
        
    }

    private Vector3 GetPosition(int index)
    {
        Vector3 pos = Vector3.zero;
        var binary = Convert.ToString(index, 2).PadLeft(3, '0');
        
        pos += distance * int.Parse(binary[0].ToString()) * Vector3.right;
        pos += distance * int.Parse(binary[1].ToString()) * Vector3.up;
        pos += distance * int.Parse(binary[2].ToString()) * Vector3.forward;
        
        return pos;
    }
}