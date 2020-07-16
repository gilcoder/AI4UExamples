using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ai4u.ext;

public class Manager : MonoBehaviour
{

    public GameObject action1;
    public GameObject action2;
    public GameObject action3;
    public GameObject action4;

    public GameObject gate1Pos1;
    public GameObject gate1Pos2;
    public GameObject gate2Pos1;
    public GameObject gate2Pos2;

    public GameObject agent;
    public GameObject gate1, gate2;

    public GameObject respawn1, respawn2, respawn3;

    private Vector3 gate1Pos, gate2Pos;
    private Quaternion gate1Rot, gate2Rot;

    void Awake() {
        gate1Pos = gate1.transform.position;
        gate2Pos = gate2.transform.position;
        gate1Rot = gate1.transform.rotation;
        gate2Rot = gate2.transform.rotation;
    }

    // Start is called before the first frame update
    public void ResetGame()
    {
        int choosePosition = Random.Range(0, 4);
        int chooseRespawn = Random.Range(0, 3);
        int chooseOpenController = Random.Range(0, 2);
        gate1.transform.position = gate1Pos;
        gate2.transform.position = gate2Pos;
        gate1.transform.rotation = gate1Rot;
        gate2.transform.rotation = gate2Rot;
        action1.GetComponent<GateController>().ResetGate();
        action2.GetComponent<GateController>().ResetGate();
        action3.GetComponent<GateController>().ResetGate();
        action4.GetComponent<GateController>().ResetGate();

        if (chooseOpenController ==  0)
        {
            TouchRewardFunc touchFunc1 = action1.GetComponent<TouchRewardFunc>();
            touchFunc1.allowNext = true;
            TouchRewardFunc touchFunc2 = action2.GetComponent<TouchRewardFunc>();
            touchFunc2.allowNext = false;
            action1.GetComponent<GateController>().IsOpenController = true;
            action2.GetComponent<GateController>().IsOpenController = false;
            action3.GetComponent<GateController>().IsOpenController = false;
            action4.GetComponent<GateController>().IsOpenController = true;
        } else
        {
            TouchRewardFunc touchFunc1 = action1.GetComponent<TouchRewardFunc>();
            touchFunc1.allowNext = false;
            TouchRewardFunc touchFunc2 = action2.GetComponent<TouchRewardFunc>();
            touchFunc2.allowNext = true;
            action1.GetComponent<GateController>().IsOpenController = false;
            action2.GetComponent<GateController>().IsOpenController = true;
            action3.GetComponent<GateController>().IsOpenController = true;
            action4.GetComponent<GateController>().IsOpenController = false;
        }

        switch(chooseRespawn)
        {
            case 0:
                agent.transform.position = respawn1.transform.position;
                agent.transform.rotation = respawn1.transform.rotation;
                break;
            case 1:
                agent.transform.position = respawn2.transform.position;
                agent.transform.rotation = respawn2.transform.rotation;
                break;
            case 2:
                agent.transform.position = respawn3.transform.position;
                agent.transform.rotation = respawn3.transform.rotation;
                break;
        }

        switch (choosePosition)
        {
            case 0:
                action1.transform.position = gate1Pos1.transform.position;
                action2.transform.position = gate1Pos2.transform.position;
                action3.transform.position = gate2Pos1.transform.position;
                action4.transform.position = gate2Pos2.transform.position;
                break;
            case 1:
                action1.transform.position = gate1Pos2.transform.position;
                action2.transform.position = gate1Pos1.transform.position;
                action3.transform.position = gate2Pos1.transform.position;
                action4.transform.position = gate2Pos2.transform.position;
                break;
            case 2:
                action1.transform.position = gate1Pos1.transform.position;
                action2.transform.position = gate1Pos2.transform.position;
                action3.transform.position = gate2Pos2.transform.position;
                action4.transform.position = gate2Pos1.transform.position;
                break;
            case 3:
                action1.transform.position = gate1Pos2.transform.position;
                action2.transform.position = gate1Pos1.transform.position;
                action3.transform.position = gate2Pos2.transform.position;
                action4.transform.position = gate2Pos1.transform.position;
                break;
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
