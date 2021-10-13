using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace GamePaint
{
    public class ModelService
    {
        private static ModelService instance;
        
        private Dictionary<string, object> cachedModelOutputs; // Replace object typing with something more appropriate to image file extension/Unity once known

        private string currSearchTerm;

        private async Task<bool> _QueryModelServer(string searchInput)
        {
            Debug.Log("QueryModelServer started. Current search term: " + currSearchTerm);
            if (!cachedModelOutputs.ContainsKey(currSearchTerm)) // !cachedModelOutputs.ContainsKey(searchInput)
            {
                try
                {
                    // query model endpoint using searchInput
                    // subscribe to Observable and set image
                    await Task.Delay(3000);
                    Debug.Log("QueryModelServer finished");
                } catch (Exception)
                {
                    // handle exception
                    return false;
                }
            }

            return true;
        }

        private static ModelService Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = new ModelService();
                    instance.cachedModelOutputs = new Dictionary<string, object>();
                }

                return instance;
            }
        }

        public static async Task<bool> QueryModelServer()
        {
            return await Instance._QueryModelServer(Instance.currSearchTerm);
        }

        public static void SetCurrentSearchTerm(string searchInput)
        {
            Debug.Log("Current search term set to: " + searchInput);
            Instance.currSearchTerm = searchInput;
        }

        public static object GetModelOutput()
        {
            var currSearchTerm = Instance.currSearchTerm;
            if (!string.IsNullOrEmpty(currSearchTerm))
            {
                return Instance.cachedModelOutputs.ContainsKey(currSearchTerm) ? Instance.cachedModelOutputs[currSearchTerm] : null;
            }

            return null;
        }
    }
}