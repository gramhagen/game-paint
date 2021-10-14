using System.Collections;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace GamePaint
{
    public class FrameManager : MonoBehaviour
    {
        public ARRaycastManager arRaycastManager;
        public ARPlaneManager arPlaneManager;
        public GameObject framePrefab;
        public Canvas canvas;
        public Image img;
        // public Button resetButton;

        private bool frameCreated = false;
        private GameObject instantiatedFrameObject;

        private bool defualtFileFlag = false;
        private string defaultFilePath = "loading1.jpg";
        private string homeFilePath = "sampleImage.jpg";


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
                        //Rraycast Planes
                        if (arRaycastManager.Raycast(touch.position, arRaycastHits))
                        {
                            // Creates or Moves the frame to the target location
                            var pose = arRaycastHits[0].pose;
                            if (frameCreated)
                            {
                                instantiatedFrameObject.transform.position = pose.position;
                                flipImage();
                            }
                            else
                            {
                                CreateCube(pose.position);
                                flipImage();
                                return;
                            }
                        }

                        // This has never deleted the cube? ignore
                        Ray ray = Camera.main.ScreenPointToRay(touch.position);
                        if (Physics.Raycast(ray, out RaycastHit hit))
                        {
                            if (hit.collider.tag == "frame")
                            {
                                DeleteCube(hit.collider.gameObject);
                            }
                        }
                    }
                }
            }
        }



        // This tries to read a file, convert to texture/material and apply it to the GameObject. Currently uses a hardcoded path (to the Loading image"
        private void flipImage()
        {
            string url = "";
            string resourcePath = "";
            if (defualtFileFlag)
            {
                url = Path.Combine(Application.streamingAssetsPath, homeFilePath);
                resourcePath = homeFilePath;
                defualtFileFlag = false;
            }
            else
            {
                url = Path.Combine(Application.streamingAssetsPath, defaultFilePath);
                resourcePath = defaultFilePath;
                defualtFileFlag = true;
            }

            //var file = Resources.Load(resourcePath);
            var tex = new Texture2D(2, 2);
            tex = (Texture2D)Resources.Load("loading1.jpg") as Texture2D;
            // for planes - instantiatedFrameObject.GetComponent<Renderer>().material.mainTexture = tex;
            //tex.LoadImage(file);

            // for cubes?
            Material myNewMaterial = new Material(Shader.Find("Standard"));
            myNewMaterial.mainTexture = tex;
            myNewMaterial.SetTexture("_MainTex", tex);
            
            instantiatedFrameObject.GetComponent<MeshRenderer>().material = myNewMaterial;

            
            /*
             * Alternate way: Using web request methods to read in the image 
            using (var uwr = UnityWebRequestTexture.GetTexture(url))
            {
                yield return uwr.SendWebRequest();

                if ((uwr.result == UnityWebRequest.Result.ConnectionError) || (uwr.result == UnityWebRequest.Result.ProtocolError))
                {
                    Debug.Log(uwr.error);
                }
                else
                {
                    // Get downloaded texture
                    var texture = DownloadHandlerTexture.GetContent(uwr);
                    //_Material.SetTexture("_MainTex", texture);
                    //Find the Standard Shader
                    Material myNewMaterial = new Material(Shader.Find("Standard"));
                    //Set Texture on the material
                    myNewMaterial.SetTexture("_MainTex", texture);
                    myNewMaterial.mainTexture = texture;

                    instantiatedFrameObject.GetComponent<MeshRenderer>().material = myNewMaterial;
                }
            }
            */
        }


        // Ignore, not used. Used to set the RawImage component as per the tutorial
        private void CreateImage(Vector3 position)
        {
            instantiatedFrameObject = Instantiate(framePrefab, Vector3.zero, Quaternion.identity);
            instantiatedFrameObject.transform.SetParent(canvas.transform, false);
            // instantiatedFrameObject.position = position;
            // instantiatedFrameObject.rectTransform.anchoredPosition = Vector3.zero;
        }

        // Creates the Cube/Plane frame from the chosen Prefab
        private void CreateCube(Vector3 position)
        {
            instantiatedFrameObject = Instantiate(framePrefab, position, Quaternion.identity);
            //instantiatedFrameObject.texture
            frameCreated = true;
            // resetButton.gameObject.SetActive(true);
        }

        // Ignore, not used. Used to pick image from user's gallery as per tutorial. not working
        private void PickImage()
        {
            NativeGallery.GetImageFromGallery(HandleMediaPickCallback, "Pick Image for the AR Frame");
        }

        // Ignore, not used. Used to pick image from user's gallery as per tutorial. not working
        private void HandleMediaPickCallback(string path)
        {
            Texture2D image = NativeGallery.LoadImageAtPath(path);
            instantiatedFrameObject.GetComponentInChildren<RawImage>().texture = image;
        }

        // Ignore, not used. Used to stop/start plane detection. Was buggy 
        private void TogglePlaneDetection(bool state)
        {
            foreach (var plane in arPlaneManager.trackables)
            {
                plane.gameObject.SetActive(state);
            }
            arPlaneManager.enabled = state;
        }

        // Ignore, not used
        public void DeleteCube(GameObject cubeObject)
        {
            Destroy(cubeObject);
            // resetButton.gameObject.SetActive(false);
            frameCreated = false;
            // TogglePlaneDetection(true);
        }
    }
}