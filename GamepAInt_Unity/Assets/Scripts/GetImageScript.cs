using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;

public class GetImageScript : MonoBehaviour
{
    const string SERVER_URL = "http://game-paint-server.southcentralus.cloudapp.azure.com:8000/";
    const string SERVER_TOKEN = "910350ecee704db58c6a8abe6bb96fb1";
    const int WAIT_DURATION = 15;
    const int WAIT_REPEAT = 4;

    public GameObject inputField;

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

        StartCoroutine(GetImage(prompt));
    }

    IEnumerator GetImage(string prompt)
    {
        var predictRequest = new UnityWebRequest(SERVER_URL + "predict", "POST");
        TextPrompt predictPrompt = new TextPrompt(prompt);
        yield return Post(predictRequest, JsonUtility.ToJson(predictPrompt));
        var result = predictRequest.downloadHandler.text;
        Debug.Log("Received output: " + result);
        ImageRef imageRef = ImageRef.FromJson(result);

        Debug.Log(imageRef.image_id);
    }

    //// Spin object until results are ready
    //for (int i = 0; i < WAIT_REPEAT; i++)
    //{
    //    transform.Rotate(new Vector3(90, 0, 0), Space.World);
    //    yield return new WaitForSecondsRealtime(WAIT_DURATION);
    //}

    //WWWForm requestForm = new WWWForm();
    //requestForm.AddField("image_id", image_ref);
    //var retrieveRequest = UnityWebRequest.Post(SERVER_URL + "retrieve", requestForm);
    //retrieveRequest.SetRequestHeader("Content-type", "application/json");
    //retrieveRequest.SetRequestHeader("token", SERVER_TOKEN);
    //yield return retrieveRequest.SendWebRequest();
    //var data = retrieveRequest.downloadHandler.text;
}
