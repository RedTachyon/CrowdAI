using System;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;


public class Boards : MonoBehaviour
{

    public float distance = 20;
    
    [Range(1, 27)]  // Careful - anything beyond like 16 is a massive stretch
    public int boards = 1;

    private const int Dims = 3;
    private void Awake()
    {
        // var board = GetComponentInChildren<Transform>();
        var board = transform.GetChild(0);
        // board.localPosition = GetPosition(1);
        
        // Debug.Log(board.name);

        Debug.Log("Cloning boards");
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

        
        var layoutBase = Mathf.CeilToInt(Mathf.Pow(boards, 1f / Dims));
        var repr = MLUtils.DecimalToArbitrarySystem(index, layoutBase).PadLeft(Dims, '0');

        
        pos += distance * int.Parse(repr[0].ToString()) * Vector3.up;
        pos += distance * int.Parse(repr[1].ToString()) * Vector3.forward;
        pos += distance * int.Parse(repr[2].ToString()) * Vector3.right;
        
        return pos;
    }
}