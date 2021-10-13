using System.Collections;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace GamePaint
{
    public class UIControls : MonoBehaviour
    {

        public async Task Search()
        {
            if (await ModelService.Query("Dog"))
            {
                SceneManager.LoadScene("ARArtScene");
            }
        }

        public void MainMenu()
        {
            SceneManager.LoadScene("UI");
        }

        public void TestDummyMethod()
        {
            SceneManager.LoadScene("ARArtScene");
        }
    }
}

