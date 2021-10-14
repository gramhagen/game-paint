using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace GamePaint
{
    public class ProcessTextInput : MonoBehaviour
    {
        public Slider progressBar;

        void Start()
        {
            var input = gameObject.GetComponent<InputField>();
            input.onEndEdit.AddListener(SubmitText);
        }

        private void SubmitText(string text)
        {
            // TODO search term validation
            ModelService.SetCurrentSearchTerm(text);
            ModelService.SetLoadingBar(progressBar);
        }
    }
}
