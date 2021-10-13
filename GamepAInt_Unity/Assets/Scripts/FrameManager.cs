using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.UI;


public class FrameManager : MonoBehaviour
{
    public ARRaycastManager arRaycastManager;
    public ARPlaneManager arPlaneManager;
    public GameObject framePrefab;
    // public Button resetButton;

    private bool frameCreated = false;
    private GameObject instantiatedFrameObject;

    private List<ARRaycastHit> arRaycastHits = new List<ARRaycastHit>();

    /*
    private void Awake()
    {
        resetButton.onClick.RemoveAllListeners();
        resetButton.onClick.AddListener(() =>
        {
            DeleteCube(instantiatedFrameObject);
        });
    }
    */

    void Update()
    {
        if (Input.touchCount > 0)
        {
            var touch = Input.GetTouch(0);
            if (touch.phase == TouchPhase.Ended)
            {
                if (Input.touchCount == 1)
                {
                    if (!frameCreated)
                    {
                        //Rraycast Planes
                        if (arRaycastManager.Raycast(touch.position, arRaycastHits))
                        {
                            var pose = arRaycastHits[0].pose;
                            CreateCube(pose.position);
                            TogglePlaneDetection(false);
                            return;
                        }
                    }

                    Ray ray = Camera.main.ScreenPointToRay(touch.position);
                    if (Physics.Raycast(ray, out RaycastHit hit))
                    {
                        if (hit.collider.tag == "frame")
                        {
                            PickImage();
                        }
                    }
                }
            }
        }
    }

    private void CreateCube(Vector3 position)
    {
        instantiatedFrameObject = Instantiate(framePrefab, position, Quaternion.identity);
        frameCreated = true;
        // resetButton.gameObject.SetActive(true);
    }

    private void PickImage()
    {
        NativeGallery.GetImageFromGallery(HandleMediaPickCallback, "Pick Image for the AR Frame");
    }

    private void HandleMediaPickCallback(string path)
    {
        Texture2D image = NativeGallery.LoadImageAtPath(path);
        instantiatedFrameObject.GetComponentInChildren<RawImage>().texture = image;
    }

    private void TogglePlaneDetection(bool state)
    {
        foreach (var plane in arPlaneManager.trackables)
        {
            plane.gameObject.SetActive(state);
        }
        arPlaneManager.enabled = state;
    }


    public void DeleteCube(GameObject cubeObject)
    {
        Destroy(cubeObject);
        // resetButton.gameObject.SetActive(false);
        frameCreated = false;
        TogglePlaneDetection(true);
    }
}