using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace GamePaint
{
    public class ModelService : MonoBehaviour
    {
        private static ModelService instance = new ModelService(); // singleton

        private Dictionary<string, object> cachedModelOutputs; // Replace object typing with something more appropriate to image file extension/Unity once known

        private ModelService()
        {
            instance.cachedModelOutputs = new Dictionary<string, object>();
        }

        private async Task<bool> QueryModelServer(string searchInput)
        {
            if (!cachedModelOutputs.ContainsKey(searchInput))
            {
                try
                {
                    // query model endpoint using searchInput
                    // subscribe to Observable and set image
                    await Task.Delay(3000);
                } catch (Exception e)
                {
                    // handle exception
                    return false;
                }
            }

            return true;
        }

        public static async Task<bool> Query(string searchInput)
        {
            return await instance.QueryModelServer(searchInput);
        }

        public static object GetModelOutput(string searchInput)
        {
            return instance.cachedModelOutputs.ContainsKey(searchInput) ? instance.cachedModelOutputs[searchInput] : null;
        }
    }
}