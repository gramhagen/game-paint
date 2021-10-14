using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.UI;
using UnityEngine.Networking;

[RequireComponent(typeof(ARRaycastManager))]
public class TapToPlaceImage : MonoBehaviour
{
    public GameObject canvasObject;

    private RawImage spawnedImage;

    private ARRaycastManager _arRaycastManager;

    private Vector2 touchPosition;

    static List<ARRaycastHit> hits = new List<ARRaycastHit>();

    // Start is called before the first frame update
    void Awake()
    {
        _arRaycastManager = GetComponent<ARRaycastManager>();

    }

    bool GetTouchPosition(out Vector2 touchPosition)
    {
        if (Input.touchCount > 0)
        {
            touchPosition = Input.GetTouch(0).position;
            return true;
        }

        touchPosition = default;
        return false;
    }

    // Update is called once per frame
    void Update()
    {
        if (!GetTouchPosition(out Vector2 touchPosition))
            return;
  
        if (_arRaycastManager.Raycast(touchPosition, hits, TrackableType.PlaneWithinPolygon))
        {
            var hitPose = hits[0].pose;

            spawnedImage = canvasObject.GetComponentInChildren<RawImage>();
            spawnedImage.transform.position = hitPose.position;

            // if (spawnedImage == null)
            //{
            //    spawnedImage = Instantiate(canvasObject, hitPose.position, hitPose.rotation);
            //}
            //else
            //{
            //    spawnedImage.transform.position = hitPose.position;
            //}
        }
    }
}
