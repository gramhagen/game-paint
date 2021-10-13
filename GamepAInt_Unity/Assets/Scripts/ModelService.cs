using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace GamePaint
{
    public class ModelService : MonoBehaviour
    {
        private Dictionary<string, object> cachedModelOutputs; // Replace object typing with something more appropriate to image file extension/Unity once known

        public static async Task<bool> QueryModelServer(string searchInput)
        {
            if (true) // !cachedModelOutputs.ContainsKey(searchInput)
            {
                try
                {
                    // query model endpoint using searchInput
                    // subscribe to Observable and set image
                    await Task.Delay(3000);
                } catch (Exception)
                {
                    // handle exception
                    return false;
                }
            }

            return true;
        }

        public static object GetModelOutput(string searchInput)
        {
            // connect to cached store somehow, singleton pattern did not work http://www.unitygeek.com/unity_c_singleton/
            return null;
        }
    }
}