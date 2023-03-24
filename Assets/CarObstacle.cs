using System;
using System.Collections;
using System.Collections.Generic;
using Managers;
using UnityEngine;

public class CarObstacle : MonoBehaviour
{
    private Rigidbody _rigidbody;

    private Vector3 originalPosition;
    // Start is called before the first frame update
    void Start()
    {
        _rigidbody = GetComponent<Rigidbody>();
        originalPosition = transform.position;
    }

    private void FixedUpdate()
    {
        if (Manager.Instance.Timestep == 2)
        {
            Reset();
        }
        _rigidbody.velocity = new Vector3(0, 0, 1f);
    }

    private void Reset()
    {
        transform.position = originalPosition;
    }
}
