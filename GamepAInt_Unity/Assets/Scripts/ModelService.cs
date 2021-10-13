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

        private async Task<bool> QQueryModelServer(string searchInput)
        {
            Debug.Log("bzl: QueryModelServer started");
            if (true) // !cachedModelOutputs.ContainsKey(searchInput)
            {
                try
                {
                    // query model endpoint using searchInput
                    // subscribe to Observable and set image
                    await Task.Delay(3000);
                    Debug.Log("bzl: QueryModelServer finished");
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

        public static async Task<bool> QueryModelServer(string searchInput)
        {
            return await Instance.QQueryModelServer(searchInput);
        }

        public static object GetModelOutput(string searchInput)
        {
            return Instance.cachedModelOutputs.ContainsKey(searchInput) ? Instance.cachedModelOutputs[searchInput] : null;
        }
    }
}