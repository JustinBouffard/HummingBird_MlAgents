using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEditor.ShaderGraph.Internal;
using UnityEngine;

/// <summary>
/// A hummingbird machine learning agent
/// </summary>
public class HummingBirdAgent : Agent
{
    [Tooltip("Force to apply when moving")]
    public float moveForce = 2f;

    [Tooltip("Speed to pitch up or down")]
    public float pitchSpeed = 100f;

    [Tooltip("Speed to rotate around the up axis")]
    public float yawSpeed = 100f;

    [Tooltip("Transform at the tip of the beak")]
    public Transform beakTip;

    [Tooltip("The agent's camera")]
    public Camera agentCamera;

    [Tooltip("Wheter this is training mode or gameplay mode")]
    public bool trainingMode;

    // The rigidbody of the agent
    new private Rigidbody rigidbody;

    //The flower area that the agent is in
    private FlowerArea flowerArea;

    // The nearest flower to the agent
    private Flower nearestFlower;

    // Allows for smoother pitch changes
    private float smoothPitchChange = 0f;

    // Allows for smoother yaw changes
    private float smoothYawChange = 0f;

    // Maximum angle that the bird can pitch up or down
    private const float MaxPitchAngle = 80f;

    // Maximum distance from the beak tip to accept nectar collision
    private const float BeakTipRadius = 0.008f;

    // Whether the agent is frozen (not flying)
    private bool frozen = false;

    /// <summary>
    /// Amount of nectar the agent has obtained this episode
    /// </summary>
    public float NectarObtained { get; private set; }

    /// <summary>
    /// Initialize the agent
    /// </summary>
    public override void Initialize()
    {
        rigidbody = GetComponent<Rigidbody>();
        flowerArea = GetComponentInParent<FlowerArea>();

        // If not training mode, no next step, play forever
        if (!trainingMode) MaxStep = 0;
    }

    /// <summary>
    /// Reset the agent when an episode begins
    /// </summary>
    public override void OnEpisodeBegin()
    {
        if(trainingMode)
        {
            // Only reset flowers in training when there is one agent per area
            flowerArea.ResetFlowers();
        }

        // Reset nectar obtained
        NectarObtained = 0f;

        // Zero out velocities so that movement stops before a new episode begins
        rigidbody.velocity = Vector3.zero;
        rigidbody.angularVelocity = Vector3.zero;

        // Default to spawning in front of a flower
        bool inFrontOfFlower = true;
        if(trainingMode)
        {
            // Spawn in front of flower 50% of the time
            inFrontOfFlower = UnityEngine.Random.value > .5f;
        }

        // Move the agent to a new random position
        MoveToSafeRandomPosition(inFrontOfFlower);

        // Recalculate the nearest flower now that the agent has moved
        UpdateNearestFlower();
    }

    /// <summary>
    /// Called when an action is received from either the player input or the neural network 
    /// 
    /// vectorActions[i] represents:
    /// Index 0 : move vector x (+1 = right, -1 = left)
    /// Index 1 : move vector y (+1 = up, -1 = down)
    /// Index 2 : move vector z (+1 = foward, -1 = backward)
    /// Index 3 : pitch angle (+1 = pitch up, -1 = pitch down)
    /// Index 4 : yaw angle (+1 = turn right, -1 = turn left)
    /// </summary>
    /// <param name="actions">The actions to take</param>
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Don't take actions if froze
        if (frozen) return;

        // Extract the vectorActions from the ActionBuffers
        ActionSegment<float> vectorActions = actions.ContinuousActions;

        // Calculate movement vector
        Vector3 move = new Vector3(vectorActions[0], vectorActions[1], vectorActions[2]);

        // Add force in the direction of the move vector
        rigidbody.AddForce(move * moveForce);

        // Get the current rotation
        Vector3 rotationVector = transform.rotation.eulerAngles;

        // Calculate pitch and yaw rotation
        float pitchChange = vectorActions[3];
        float yawChange = vectorActions[4];

        // Calculate smooth rotation changes
        smoothPitchChange = Mathf.MoveTowards(smoothPitchChange, pitchChange, 2f * Time.fixedDeltaTime);
        smoothYawChange = Mathf.MoveTowards(smoothYawChange, yawChange, 2f * Time.fixedDeltaTime);

        // Calculate new pitch and yaw based on smooth values
        // Clamp pitch to avoid flipping upside down
        float pitch = rotationVector.x + smoothPitchChange * Time.fixedDeltaTime * pitchSpeed;
        if (pitch > 180f) pitch -= 360f;
        pitch = Mathf.Clamp(pitch, -MaxPitchAngle, MaxPitchAngle);

        float yaw = rotationVector.y + smoothYawChange * Time.fixedDeltaTime * yawSpeed;

        // Apply the new rotation
        transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
    } 

    /// <summary>
    /// Collect vector observations from the environment
    /// </summary>
    /// <param name="sensor">The vector sensor</param>
    public override void CollectObservations(VectorSensor sensor)
    {
        if (nearestFlower == null)
        {
            sensor.AddObservation(new float[10]);
            return;
        }

        // Observe the agent's local rotation (4 obeservations)
        sensor.AddObservation(transform.localRotation.normalized);

        // Get a vector from the beak tip to the nearest flower
        Vector3 toFlower = nearestFlower.flowerCenterPosition - beakTip.position;

        // Observe a normalized vector pointing to the nearest flower (3 observations)
        sensor.AddObservation(toFlower.normalized);

        // Observe a dot product that indicates whether the beak tip si in front of the flower (1 observation)
        // (+ 1 means that the beak tip is directly in front of the flower, -1 means directly behind)
        sensor.AddObservation(Vector3.Dot(toFlower, -nearestFlower.FlowerUpVector.normalized));

        // Observe a dot product that indicates whether the beak is pointing towards the flower (1 observation)
        // (+1 means that the beak is pointing direcly at the flower, -1 means directly away)
        sensor.AddObservation(Vector3.Dot(beakTip.forward.normalized, -nearestFlower.FlowerUpVector.normalized));

        // Observe the relative distance from the beak tip to the flower (1 observation)
        sensor.AddObservation(toFlower.magnitude / FlowerArea.AreaDiameter);

        // 10 total observations
    }

    /// <summary>
    /// When behavior type is set to heuristic only on the agent's behavior parameters
    /// this function will be called. it's return values will be feed into
    /// <see cref="OnActionReceived(ActionBuffers)"/> instead of using the neural network
    /// </summary>
    /// <param name="actionsOut">An output action array</param>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Create placeholders for all movement/turning
        Vector3 foward = Vector3.zero;
        Vector3 left = Vector3.zero;
        Vector3 up = Vector3.zero;
        float pitch = 0f;
        float yaw = 0f;

        // Convert inputs to movment and turning
        // All values should be between 1 and -1

        // Foward/backward
        if (Input.GetKey(KeyCode.W)) foward = transform.forward;
        else if (Input.GetKey(KeyCode.S)) foward = -transform.forward;


        // Left/Right
        if (Input.GetKey(KeyCode.A)) left = -transform.right;
        else if (Input.GetKey(KeyCode.D)) left = transform.right;

        // Up/Down
        if (Input.GetKey(KeyCode.E)) up = transform.up;
        else if (Input.GetKey(KeyCode.Q)) up = -transform.up;

        // Pitch up/Down
        if (Input.GetKey(KeyCode.UpArrow)) pitch = -1f;
        else if (Input.GetKey(KeyCode.DownArrow)) pitch = 1f;

        // Turn left/Right
        if (Input.GetKey(KeyCode.LeftArrow)) yaw = -1f;
        else if (Input.GetKey(KeyCode.RightArrow)) yaw = 1f;

        // Combine the movement vectors and normalize
        Vector3 combined = (foward + left + up).normalized;

        // Extract the actions from the ActionBuffers
        ActionSegment<float> actions = actionsOut.ContinuousActions;

        // Add the 3 movement values, pitch and yaw to the actions out array
        actions[0] = combined.x;
        actions[1] = combined.y;
        actions[2] = combined.z;
        actions[3] = pitch;
        actions[4] = yaw;
    }

    /// <summary>
    /// Prevent agent from moving and taking actions
    /// </summary>
    public void FreezeAgent()
    {
        UnityEngine.Debug.Assert(trainingMode == false, "Freeze/Unfreeze not supported in training");
        frozen = true;
        rigidbody.Sleep();
    }

    /// <summary>
    /// Resume agent movement and actions
    /// </summary>
    public void UnfreezeAgent()
    {
        UnityEngine.Debug.Assert(trainingMode == false, "Freeze/Unfreeze not supported in training");
        frozen = false;
        rigidbody.WakeUp();
    }


    /// <summary>
    /// Move the agent to a safe random position (i.e does not collide with anything)
    /// If in front of flower also point the beak at the flower
    /// </summary>
    /// <param name="inFrontOfFlower">Whether to choose a spot in front of a flower</param>
    private void MoveToSafeRandomPosition(bool inFrontOfFlower)
    {
        bool safePositionFound = false;
        int attemptsRemaining = 100;    //Prevent an infinite loop
        Vector3 potentialPosition = Vector3.zero;
        Quaternion potentialRotation = new Quaternion();

        // Loop until a safe position is found or we run out of options
        while(!safePositionFound && attemptsRemaining > 0)
        {
            attemptsRemaining--;
            if(inFrontOfFlower)
            {
                // Pick a random flower
                Flower randomFlower = flowerArea.Flowers[UnityEngine.Random.Range(0, flowerArea.Flowers.Count)];

                // Position 10 to 20 cm in front of the flower
                float distanceFromFlower = UnityEngine.Random.Range(.1f, .2f);
                potentialPosition = randomFlower.transform.position + randomFlower.FlowerUpVector * distanceFromFlower;

                // Point beak at flower (bird head is center of transform)
                Vector3 toFlower = randomFlower.flowerCenterPosition - potentialPosition;
                potentialRotation = Quaternion.LookRotation(toFlower, Vector3.up);
            }
            else
            {
                // Pick a random height from the ground
                float height = UnityEngine.Random.Range(1.2f, 2.5f);

                // Pick a random radius from the center of the area
                float radius = UnityEngine.Random.Range(2f, 7f);

                // Pick a random direction rotated around the y axis
                Quaternion direction = Quaternion.Euler(0f, UnityEngine.Random.Range(-100f, 100f), 0f);

                // Combine height, radius, and direction to pick a potential position
                potentialPosition = flowerArea.transform.position + Vector3.up * height + direction * Vector3.forward * radius;


                // Choose and set random starting pitch and yaw
                float pitch = UnityEngine.Random.Range(-60f, 60f);
                float yaw = UnityEngine.Random.Range(-180f, 180f);
                potentialRotation = Quaternion.Euler(pitch, yaw, 0f);
            }

            // Check to see if the agent will collide with anything
            Collider[] colliders = Physics.OverlapSphere(potentialPosition, 0.05f);

            // Safe position has been found if no colliders are overlaped
            safePositionFound = colliders.Length == 0;
        }

        UnityEngine.Debug.Assert(safePositionFound, "Could not find a safe position to spawn");

        // Set the position and rotation
        transform.position = potentialPosition;
        transform.rotation = potentialRotation;
    }

    /// <summary>
    /// Update the nearest flower to the agent
    /// </summary>
    private void UpdateNearestFlower()
    {
        foreach(Flower flower in flowerArea.Flowers)
        {
            if(nearestFlower == null && flower.HasNectar)
            {
                // No current nearest flower and this flower has nectar, so set this flower
                nearestFlower = flower;
            }
            else if(flower.HasNectar)
            {
                // Calculate distance to this flower and distance to the current nearest flower
                float distanceToFlower = Vector3.Distance(flower.transform.position, beakTip.position);
                float distanceToCurrentNearestFlower = Vector3.Distance(nearestFlower.transform.position, beakTip.position);

                // If current nearest flower is empty or this flower is closer, update nearest flower
                if(!nearestFlower.HasNectar || distanceToFlower < distanceToCurrentNearestFlower)
                {
                    nearestFlower = flower;
                }
            }
        }
    }

    /// <summary>
    /// Called when the agent's collider enters a trigger collider
    /// </summary>
    /// <param name="other">The trigger collider</param>
    private void OnTriggerEnter(Collider other)
    {
        TriggerEnterOrStay(other);
    }

    /// <summary>
    /// Called when the agent's collider stays in a trigger collider
    /// </summary>
    /// <param name="other">The trigger collider</param>
    private void OnTriggerStay(Collider other)
    {
        TriggerEnterOrStay(other);
    }

    /// <summary>
    /// Handles when the agent's collider enters or stays in a trigger collider
    /// </summary>
    /// <param name="collider"></param>
    private void TriggerEnterOrStay(Collider collider)
    {
        // Check if agent is colliding with nectar
        if (collider.CompareTag("nectar"))
        {
            Vector3 closestPointToBeakTip = collider.ClosestPoint(beakTip.position);

            // Check if the closest collision point is close to the beak tip
            // Note: a collision with anything but the beak tip should not count
            if (Vector3.Distance(beakTip.position, closestPointToBeakTip) < BeakTipRadius)
            {
                // Look up the flower for this nectar collider
                Flower flower = flowerArea.GetFlowerFromNectar(collider);

                // Attempt to take .01 nectar
                // Note: this is fixed timestep, meaning it happens 50 times per second
                float nectarReceived = flower.Feed(0.01f);

                // Keep track of nectar obtained 
                NectarObtained += nectarReceived;

                if (trainingMode)
                {
                    // Calculate reward for getting nectar
                    float bonus = .02f * Mathf.Clamp01(Vector3.Dot(transform.forward.normalized, -nearestFlower.FlowerUpVector.normalized));
                    AddReward(.01f + bonus);
                }

                // If flower is empty, update the nearest flower
                if (!flower.HasNectar)
                {
                    UpdateNearestFlower();
                }
            }
        }
    }

    /// <summary>
    /// Called when the agent collides with something solid
    /// </summary>
    /// <param name="collision">The collision info</param>
    private void OnCollisionEnter(Collision collision)
    {
        if (trainingMode && collision.collider.CompareTag("boundary"))
        {
            // Collided with the area boundary, give a negative reward
            AddReward(-.5f);
        }
    }

    /// <summary>
    /// Called every frame
    /// </summary>
    private void Update()
    {
        // Draw a line from the beak tip to the nearest flowerà
        if (nearestFlower != null)  UnityEngine.Debug.DrawLine(beakTip.position, nearestFlower.flowerCenterPosition, Color.red);
    }

    /// <summary>
    /// Called every .02 seconds
    /// </summary>
    private void FixedUpdate()
    {
        // Avoid senarie where nearest flower nectar is stolen by opponent and not updated
        if (nearestFlower != null && !nearestFlower.HasNectar) UpdateNearestFlower();
    }
}
