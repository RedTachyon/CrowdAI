using System.IO;
using UnityEditor;
using UnityEditor.Callbacks;
using UnityEngine;

namespace Editor
{
    public class MyBuildPostprocessor
    {

        [PostProcessBuild(1)]
        public static void OnPostprocessBuild(BuildTarget target, string pathToBuiltProject)
        {
            string projectPath;
            if (target == BuildTarget.StandaloneOSX)
            {
                projectPath = pathToBuiltProject;
            } else
            {
                projectPath = Path.GetDirectoryName(pathToBuiltProject);
            }
            var mapDirPath = Directory.GetParent(Application.dataPath)?.FullName;
        
        
            if (mapDirPath != null && projectPath != null)
            {
                var pathToMapsFolder = Path.Combine(mapDirPath, "data");

                if (!Directory.Exists(pathToMapsFolder))
                {
                    Debug.Log("Data folder not found.");
                    return;
                }
            
                var newMapPath = Path.Combine(projectPath, "data"); // TODO: Fix this on Mac
                Debug.Log($"Data path: {pathToMapsFolder}. Copying data from there to the built project.");
                DirectoryCopy(pathToMapsFolder, newMapPath, true);
            }

        }

        private static void DirectoryCopy(string sourceDirName, string destDirName, bool copySubDirs)
        {
            // Get the subdirectories for the specified directory.
            // Source: https://docs.microsoft.com/en-us/dotnet/standard/io/how-to-copy-directories
            var dir = new DirectoryInfo(sourceDirName);

            if (!dir.Exists)
            {
                throw new DirectoryNotFoundException(
                    "Source directory does not exist or could not be found: "
                    + sourceDirName);
            }

            var dirs = dir.GetDirectories();

            // If the destination directory doesn't exist, create it.       
            Directory.CreateDirectory(destDirName);

            // Get the files in the directory and copy them to the new location.
            var files = dir.GetFiles();
            foreach (var file in files)
            {
                var tempPath = Path.Combine(destDirName, file.Name);
                file.CopyTo(tempPath, false);
            }

            // If copying subdirectories, copy them and their contents to new location.
            if (copySubDirs)
            {
                foreach (var subdir in dirs)
                {
                    var tempPath = Path.Combine(destDirName, subdir.Name);
                    DirectoryCopy(subdir.FullName, tempPath, copySubDirs);
                }
            }
        }

    }
}