using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;

public class GetImageScript : MonoBehaviour
{
    const string SERVER_URL = "http://game-paint-server.southcentralus.cloudapp.azure.com:8000/";
    const string SERVER_TOKEN = "910350ecee704db58c6a8abe6bb96fb1";
    const int WAIT_SECONDS = 60;

    public GameObject inputField;
    public RawImage image;
    public Slider loadingBar;
    public string outputPath;

    [Serializable]
    public class TextPrompt
    {
        public string prompt;
        public TextPrompt(string prompt)
        {
            this.prompt = prompt;
        }
    }

    [Serializable]
    public class ImageRef
    {
        public string image_id;
        public ImageRef(string image_id)
        {
            this.image_id = image_id;
        }
        public static ImageRef FromJson(string jsonString)
        {
            return JsonUtility.FromJson<ImageRef>(jsonString);
        }
    }

    IEnumerator Post(UnityWebRequest request, string json)
    {
        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
        request.uploadHandler = (UploadHandler) new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = (DownloadHandler) new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");
        request.SetRequestHeader("token", SERVER_TOKEN);
        yield return request.SendWebRequest();
        Debug.Log("Status Code: " + request.responseCode);
    }

    public void onClick()
    {
        string prompt = inputField.GetComponent<InputField>().text;
        Debug.Log("Received input: " + prompt);

        loadingBar.GetComponent<CanvasGroup>().alpha = 1;
        loadingBar.value = 0f;
        StartCoroutine(GetImage(prompt));
    }

    IEnumerator GetImage(string prompt)
    {
        var predictRequest = new UnityWebRequest(SERVER_URL + "predict", "POST");
        TextPrompt predictPrompt = new TextPrompt(prompt);
        yield return Post(predictRequest, JsonUtility.ToJson(predictPrompt));
        var predictResult = predictRequest.downloadHandler.text;
        Debug.Log("Predict output: " + predictResult);
        ImageRef imageRef = ImageRef.FromJson(predictResult);

        Debug.Log(imageRef.image_id);

        for (int i = 1; i <= WAIT_SECONDS; i++)
        {
            yield return new WaitForSecondsRealtime(1);
            loadingBar.value = 0.9f * i / WAIT_SECONDS;
        }
        loadingBar.GetComponent<CanvasGroup>().alpha = 0;

        var retrieveRequest = new UnityWebRequest(SERVER_URL + "retrieve", "POST");
        yield return Post(retrieveRequest, JsonUtility.ToJson(imageRef));
        var retrieveResult = retrieveRequest.downloadHandler.data;
        Debug.Log("Retrieve output: " + retrieveResult);

        outputPath = Application.dataPath + "/../output.png";
        File.WriteAllBytes(outputPath, retrieveResult);

        Texture2D texture = new Texture2D(2, 2);
        bool loaded = texture.LoadImage(retrieveResult);
        if (loaded)
        {
            image.texture = texture;
        }
    }
}
