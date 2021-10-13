using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace GamePaint
{
    public class ProcessTextInput : MonoBehaviour
    {
        void Start()
        {
            var input = gameObject.GetComponent<InputField>();
            input.onEndEdit.AddListener(SubmitText);
        }

        private void SubmitText(string arg0)
        {
            // TODO search term validation
            ModelService.SetCurrentSearchTerm(arg0);
        }
    }
}
