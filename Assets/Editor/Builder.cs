using UnityEditor;
using UnityEngine;

namespace Editor
{
    class Builder
    {
        static void PerformBuild ()
        {

            var osString = SystemInfo.operatingSystem;
            BuildTarget target;

            if (osString.StartsWith("Windows"))
            {
                target = BuildTarget.StandaloneWindows;
            }
            else if (osString.StartsWith("Mac"))
            {
                target = BuildTarget.StandaloneOSX;
            }
            else
            {
                target = BuildTarget.StandaloneLinux64;
            }
        
            BuildPlayerOptions buildPlayerOptions = new BuildPlayerOptions();
            buildPlayerOptions.scenes = new[] { "Assets/Scenes/GeneralScene.unity" };
            buildPlayerOptions.locationPathName = "builds/crowd.app";
            buildPlayerOptions.target = target;
            buildPlayerOptions.options = BuildOptions.None;
        
        
            BuildPipeline.BuildPlayer(buildPlayerOptions);
        }
    }
}