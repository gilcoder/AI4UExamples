using UnityEngine;
using UnityEngine.UI;
using System.Net.Sockets;
using System.Net;
using System.Text;
using UnityStandardAssets.Characters.ThirdPerson;
using UnityEngine.SceneManagement;
using unityremote;

namespace unityremote
{

    public class PlayerRemoteAgent : Agent
    {


        //BEGIN::Game controller variables
        private ThirdPersonCharacter character;
        private Transform m_CamTransform;
        private Vector3 m_CamForward;             // The current forward direction of the camera
        private Vector3 m_Move;
        //END::

        //BEGIN::motor controll variables
        private static float fx, fy;
        public string remoteIPAddress;
        private float speed = 0.0f;
        private bool crouch;
        private bool jump;
        private float leftTurn = 0;
        private float rightTurn = 0;
        private float up = 0;
        private float down = 0;
        private bool pushing;
        private bool resetState;
        private bool getpickup;
        private bool walkspeed;
        private UdpClient socket;
        private bool commandReceived;
        public int rayCastingWidth;
        public int rayCastingHeight;
        //END::

        private GameObject player;
        
        private PlayerRemoteSensor sensor;

        public Camera m_camera;

        // Use this for initialization
        void Start()
        {
            
            commandReceived = false;
            ResetState();
            if (!gameObject.activeSelf)
            {
                return;
            }
            player = GameObject.FindGameObjectsWithTag("Player")[0];

            if (m_camera != null)
            {
                m_CamTransform = m_camera.transform;
            }
            else
            {
                Debug.LogWarning(
                    "Warning: no main camera found. Third person character needs a Camera tagged \"MainCamera\", for camera-relative controls.", gameObject);
                // we use self-relative controls in this case, which probably isn't what the user wants, but hey, we warned them!
            }

            // get the third person character ( this should never be null due to require component )
            character = GetComponent<ThirdPersonCharacter>();
            sensor = new PlayerRemoteSensor();
            sensor.Start(m_camera, player, this.rayCastingHeight, this.rayCastingWidth);
        }

        private void ResetState()
        {
            speed = 0.0f;
            fx = 0;
            fy = 0;
            crouch = false;
            jump = false;
            pushing = false;
            leftTurn = 0;
            rightTurn = 0;
            up = 0;
            down = 0;
        }

        public override void ApplyAction()
        {
            ResetState();
            string action = GetActionName();
            switch (action)
            {
                case "fx":
                    fx = GetActionArgAsFloat();
                    break;
                case "fy":
                    fy = GetActionArgAsFloat();
                    break;
                case "left_turn":
                    leftTurn = GetActionArgAsFloat();
                    break;
                case "right turn":
                    rightTurn = GetActionArgAsFloat();
                    break;
                case "up":
                    up = GetActionArgAsFloat();
                    break;
                case "down":
                    down = GetActionArgAsFloat();
                    break;
                case "push":
                    pushing = GetActionArgAsBool();
                    break;
                case "jump":
                    jump = GetActionArgAsBool();
                    break;
                case "crouch":
                    crouch = GetActionArgAsBool();
                    break;
                case "pickup":
                    getpickup = GetActionArgAsBool();
                    break;
            }
        }

        // Update is called once per frame
        public override void UpdatePhysics()
        {
            // read inputs
            float h = fx;
            float v = fy;


            // calculate move direction to pass to character
            if (m_CamTransform != null)
            {

                // calculate camera relative direction to move:
                m_CamForward = Vector3.Scale(m_CamTransform.forward, new Vector3(1, 0, 1)).normalized;
                m_Move = v * m_CamForward + h * m_CamTransform.right;

            }
            else
            {
                // we use world-relative directions in the case of no main camera
                m_Move = v * Vector3.forward + h * Vector3.right;
            }


            // walk speed multiplier
            if (walkspeed) {
                m_Move *= speed;
            } 

            if (resetState)
            {
                ResetState();
            }

            // pass all parameters to the character control script
            character.Move(m_Move, crouch, jump, rightTurn - leftTurn, down - up, pushing, fx, fy, getpickup);
            //character.Move(m_Move, crouch, m_Jump, h, v, pushing);
            jump = false;
            sensor.UpdateViewMatrix();
        }

        public override void UpdateState()
        {
            SetStateAsByteArray(0, "frame", sensor.updateCurrentRayCastingFrame());
            SetStateAsFloat(1, "reward", 0.0f);
            SetStateAsBool(2, "touching", false);
            SetStateAsFloat(3, "touchingvalue", 0.0f);
            SetStateAsFloat(4, "energy", 30);
        }
    }

    public class PlayerRemoteSensor
    {
        private byte[] currentFrame;
        
        private RenderTexture view;
        private Camera m_camera;

        private static Socket sock;
        private static IPAddress serverAddr;
        private static EndPoint endPoint;

        private GameObject player;
        private GameObject preLoader;

        private static int life, score;
        private static float energy;


        private int verticalResolution = 20;
        private int horizontalResolution = 20;
        private bool useRaycast = true;

        private Ray[,] raysMatrix = null;
        private int[,] viewMatrix = null;
        private Vector3 fw1 = new Vector3(), fw2 = new Vector3(), fw3 = new Vector3();

        
        public void SetCurrentFrame(byte[] cf)
        {
            this.currentFrame = cf;
        }

        // Use this for initialization
        public void Start(Camera cam, GameObject player, int rayCastingHRes, int rayCastingVRes)
        {
            this.verticalResolution = rayCastingVRes;
            this.horizontalResolution = rayCastingHRes;
            life = 0;
            score = 0;
            energy = 0;
            useRaycast = true;
            currentFrame = null;

            m_camera = cam;
            this.player = player;
            fw3 = m_camera.transform.forward;


            if (useRaycast)
            {
                if (raysMatrix == null)
                {
                    raysMatrix = new Ray[verticalResolution, horizontalResolution];
                }
                if (viewMatrix == null)
                {
                    viewMatrix = new int[verticalResolution, horizontalResolution];

                }
                for (int i = 0; i < verticalResolution; i++)
                {
                    for (int j = 0; j < horizontalResolution; j++)
                    {
                        raysMatrix[i, j] = new Ray();
                    }
                }
                currentFrame = updateCurrentRayCastingFrame();
            }    
        }



        public byte[] updateCurrentRayCastingFrame()
        {
            UpdateRaysMatrix(m_camera.transform.position, m_camera.transform.forward, m_camera.transform.up, m_camera.transform.right);
            UpdateViewMatrix();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < verticalResolution; i++)
            {
                for (int j = 0; j < horizontalResolution; j++)
                {
                    //Debug.DrawRay(raysMatrix[i, j].origin, raysMatrix[i, j].direction, Color.red);
                    sb.Append(viewMatrix[i, j]).Append(",");
                }
                sb.Append(";");
            }
            return Encoding.UTF8.GetBytes(sb.ToString().ToCharArray());
        }



        private void UpdateRaysMatrix(Vector3 position, Vector3 forward, Vector3 up, Vector3 right, float fieldOfView = 45.0f)
        {


            float vangle = 2 * fieldOfView / verticalResolution;
            float hangle = 2 * fieldOfView / horizontalResolution;

            float ivangle = -fieldOfView;

            for (int i = 0; i < verticalResolution; i++)
            {
                float ihangle = -fieldOfView;
                fw1 = (Quaternion.AngleAxis(ivangle + vangle * i, right) * forward).normalized;
                fw2.Set(fw1.x, fw1.y, fw1.z);

                for (int j = 0; j < horizontalResolution; j++)
                {
                    raysMatrix[i, j].origin = position;
                    raysMatrix[i, j].direction = (Quaternion.AngleAxis(ihangle + hangle * j, up) * fw2).normalized;
                }
            }
        }

        public void UpdateViewMatrix(float maxDistance = 500.0f)
        {
            for (int i = 0; i < verticalResolution; i++)
            {
                for (int j = 0; j < horizontalResolution; j++)
                {
                    RaycastHit hitinfo;
                    if (Physics.Raycast(raysMatrix[i, j], out hitinfo, maxDistance))
                    {
                        string objname = hitinfo.collider.gameObject.name;
                        switch (objname)
                        {
                            case "Terrain":
                                viewMatrix[i, j] = 3;
                                break;
                            case "maze":
                                viewMatrix[i, j] = -1;
                                break;
                            case "Teletransporter":
                                viewMatrix[i, j] = 4;
                                break;
                            case "GoldKey":
                                viewMatrix[i, j] = 5;
                                break;
                            default:
                                objname = hitinfo.collider.gameObject.tag;
                                if (objname.StartsWith("PickUpBad", System.StringComparison.CurrentCulture))
                                {
                                    viewMatrix[i, j] = -6;
                                }
                                else if (objname.StartsWith("PickUp", System.StringComparison.CurrentCulture))
                                {
                                    viewMatrix[i, j] = 6;
                                }
                                else
                                {
                                    viewMatrix[i, j] = 1;
                                }
                                break;
                        }
                    }
                    else
                    {
                        viewMatrix[i, j] = 0;
                    }
                }
            }
        }
    }
}