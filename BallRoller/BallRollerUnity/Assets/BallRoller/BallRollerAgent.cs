using UnityEngine;
using ai4u;
using ai4u.ext;

public class BallRollerAgent : RLAgent
{
    public GameObject target;
    private Rigidbody rBody;
    public float speed = 10;
    private bool done = false;
    private float fx, fz;

    public void Start()
    {
        rBody = GetComponent<Rigidbody>();
        done = false;
        ResetReward();
    }

    private void ResetPlayer()
    {
        done = false;
        rBody.velocity = Vector3.zero;
        rBody.angularVelocity = Vector3.zero;
        target.transform.localPosition = new Vector3(Random.value * 6 - 3, 0.5f, Random.value * 6 - 3);
        transform.localPosition = new Vector3(0, 0.5f, 0);
        fx = 0;
        fz = 0;
        ResetReward();
    }

    public override void ApplyAction()
    {
        fx = 0.0f;
        fz = 0.0f;
        switch (GetActionName())
        {
            case "fx":
                fx = GetActionArgAsFloat();
                break;
            case "fz":
                fz = GetActionArgAsFloat();
                break;
            case "restart":
                ResetPlayer();
                break;
        }
    }
    
    public override void UpdatePhysics()
    {
        if (rBody != null)
        {
            rBody.AddForce(fx * speed, 0, fz * speed);
        }
    }

    void OnCollisionStay(Collision other) {
        if (other.gameObject.name == "Target") {
            done = true;
        }
    }

    public override void boxListener(BoxRewardFunc fun) {
        done = true;
    }


    public override void UpdateState()
    {
        SetStateAsBool(0, "done", done);
        SetStateAsFloat(1, "reward", Reward);
        SetStateAsFloat(2, "tx", target.transform.localPosition.x);
        SetStateAsFloat(3, "tz", target.transform.localPosition.z);
        SetStateAsFloat(4, "vx", rBody.velocity.x);
        SetStateAsFloat(5, "vz", rBody.velocity.z);
        SetStateAsFloat(6, "x", transform.localPosition.x);
        SetStateAsFloat(7, "y", transform.localPosition.y);
        SetStateAsFloat(8, "z", transform.localPosition.z);
        ResetReward();
    }
}
