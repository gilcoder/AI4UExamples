using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraToAgent : MonoBehaviour
{

    private CameraTracker tracker;
    private KeyCode[] codes;
    
    public GameObject[] agents;

    // Start is called before the first frame update
    void Start()
    {
        tracker = gameObject.GetComponent<CameraTracker>();
        codes = new KeyCode[8];
        codes[0] = KeyCode.Alpha1;
        codes[1] = KeyCode.Alpha2;
        codes[2] = KeyCode.Alpha3;
        codes[3] = KeyCode.Alpha4;
        codes[4] = KeyCode.Alpha5;
        codes[5] = KeyCode.Alpha6;
        codes[6] = KeyCode.Alpha7;
        codes[7] = KeyCode.Alpha8;    
    }

    // Update is called once per frame
    void Update()
    {

        for (int i = 0; i < agents.Length; i++)  
        {
            if (Input.GetKey(codes[i])) 
            {
                tracker.target = agents[i];
            }
        }

    }
}
