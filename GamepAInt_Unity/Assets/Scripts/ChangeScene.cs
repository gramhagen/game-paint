using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class ChangeScene : MonoBehaviour
{

    public void Play()
    {
        //   SceneManager.LoadScene("ARArtScene");
        SceneManager.LoadScene("GetImageScene");
    }

    public void MainMenu()
    {
        SceneManager.LoadScene("UI");
    }
}
