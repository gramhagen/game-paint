using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;

namespace GamePaint
{
    public struct UnityWebRequestAwaiter : INotifyCompletion
    {
        private UnityWebRequestAsyncOperation asyncOp;
        private Action continuation;

        public UnityWebRequestAwaiter(UnityWebRequestAsyncOperation asyncOp)
        {
            this.asyncOp = asyncOp;
            continuation = null;
        }

        public bool IsCompleted { get { return asyncOp.isDone; } }

        public void GetResult() { }

        public void OnCompleted(Action continuation)
        {
            this.continuation = continuation;
            asyncOp.completed += OnRequestCompleted;
        }

        private void OnRequestCompleted(AsyncOperation obj)
        {
            continuation?.Invoke();
        }
    }

    public static class ExtensionMethods
    {
        public static UnityWebRequestAwaiter GetAwaiter(this UnityWebRequestAsyncOperation asyncOp)
        {
            return new UnityWebRequestAwaiter(asyncOp);
        }
    }

    public class ModelService
    {
        const string SERVER_URL = "http://game-paint-server.southcentralus.cloudapp.azure.com:8000/";
        const string SERVER_TOKEN = "910350ecee704db58c6a8abe6bb96fb1";
        const int WAIT_SECONDS = 55;

        private Slider progressBar;
        private static ModelService instance;
        private string currSearchTerm;
        private byte[] modelOutput;

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

        public async Task<bool> Post(UnityWebRequest request, string json)
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
            request.uploadHandler = (UploadHandler)new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.SetRequestHeader("token", SERVER_TOKEN);
            await request.SendWebRequest();
            Debug.Log("Status Code: " + request.responseCode);
            // TODO: validate return status
            return true;
        }

        private async Task<bool> _QueryModelServer(string searchInput)
        {
            Debug.Log("QueryModelServer started. Current search term: " + currSearchTerm);

            var predictRequest = new UnityWebRequest(SERVER_URL + "predict", "POST");
            TextPrompt predictPrompt = new TextPrompt(searchInput);
            await Post(predictRequest, JsonUtility.ToJson(predictPrompt));
            var predictResult = predictRequest.downloadHandler.text;
            Debug.Log("Predict output: " + predictResult);
            ImageRef imageRef = ImageRef.FromJson(predictResult);

            progressBar.value = 0f;
            var progressBarCG = progressBar.GetComponent<CanvasGroup>();
            progressBarCG.alpha = 1;
            for (int i = 1; i <= WAIT_SECONDS; i++)
            {
                await Task.Delay(1000);
                progressBar.value = 0.9f * i / WAIT_SECONDS;
            }
            progressBarCG.alpha = 0;

            var retrieveRequest = new UnityWebRequest(SERVER_URL + "retrieve", "POST");
            await Post(retrieveRequest, JsonUtility.ToJson(imageRef));
            modelOutput = retrieveRequest.downloadHandler.data;
            return true;
        }

        private static ModelService Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = new ModelService();
                }

                return instance;
            }
        }

        public static async Task<bool> QueryModelServer()
        {
            return await Instance._QueryModelServer(Instance.currSearchTerm);
        }

        public static void SetLoadingBar(Slider progressBar)
        {
            Instance.progressBar = progressBar;
        }

        public static void SetCurrentSearchTerm(string searchInput)
        {
            Debug.Log("Current search term set to: " + searchInput);
            Instance.currSearchTerm = searchInput;
        }

        public static byte[] GetModelOutput()
        {
            return Instance.modelOutput;
        }
    }
}