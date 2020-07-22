using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ai4u;

public class BallController : Controller
{
    public float speed = 0.01f;
    override public object[] GetAction()
    {

        float actionValue=0;
        string actionName="";

        if (Input.GetKey(KeyCode.W))
        {
            actionName = "fz";
            actionValue = speed;
        }

        if (Input.GetKey(KeyCode.S))
        {
            actionName = "fz";
            actionValue = -speed;
        }


        if (Input.GetKey(KeyCode.A))
        {
            actionName = "fx";
            actionValue = -speed;
        }

        if (Input.GetKey(KeyCode.D))
        {
            actionName = "fx";
            actionValue = speed;
        }

        if (Input.GetKey(KeyCode.R))
        {
            actionName = "restart";
            actionValue = -1;
        }

        if (actionName != "restart")
        {
            return GetFloatAction(actionName, actionValue);
        } else
        {
            return GetFloatAction("restart", actionValue);
        }
        
    }

    override public void NewStateEvent()
    {
        float r = GetStateAsFloat(1);
        if (r != 0) {
            Debug.Log("Reward " + r);
        }
    }
}
