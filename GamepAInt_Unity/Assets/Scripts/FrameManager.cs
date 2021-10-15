using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.UI;

namespace GamePaint
{
    public class FrameManager : MonoBehaviour
    {
        public ARRaycastManager arRaycastManager;
        public ARPlaneManager arPlaneManager;
        public GameObject framePrefab;

        private GameObject frameObject;

        private List<ARRaycastHit> arRaycastHits = new List<ARRaycastHit>();

        // Called on loop (detects any touch on the plane on screen)
        void Update()
        {
            if (Input.touchCount > 0)
            {
                var touch = Input.GetTouch(0);
                if (touch.phase == TouchPhase.Ended)
                {
                    if (Input.touchCount == 1)
                    {
                        //Raycast Planes
                        if (arRaycastManager.Raycast(touch.position, arRaycastHits))
                        {
                            // Creates or Moves the frame to the target location
                            var pose = arRaycastHits[0].pose;
                            if (frameObject == null)
                            {
                                frameObject = Instantiate(framePrefab);

                                var tex = new Texture2D(2, 2);
                                tex.LoadImage(ModelService.GetModelOutput());
                                frameObject.GetComponent<MeshRenderer>().material.SetTexture("_MainTex", tex);
                                frameObject.transform.Rotate(0.0f, 180.0f, 0.0f);
                            }
                            frameObject.transform.position = pose.position;
                        }
                    }
                }
            }
        }
    }
}