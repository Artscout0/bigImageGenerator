# Big Image Generator

A personal project with some friends to generate as big of an image (in terms of file size and pixel size)
## How to use

### Installing the required dependecies

You can use the provided `.venv` file as the enviroment file with `.venv/Scripts/activate` or create your own using the steps below.

To create your own enviroment make sure you install the required dependencies : (make sure you have your pip executable directory path in your enviroment variables.)
- ```pip install glfw```
- ```pip install PyOpenGL```
- ```pip install numpy```
- ```pip install Image```
- ```pip install tiffile```

### Making sure the code runs fast

1. Open NVIDIA Control Panel: Right-click your desktop and select "NVIDIA Control Panel."
2. Go to "Manage 3D settings": On the left-hand side, click "Manage 3D settings."
3. Select "Program Settings": Click the "Program Settings" tab.
4. Add your Python:
   - Click the "Add" button.
   - Find the exact `python.exe` or `pythonw.exe` that runs your script. If it's not in the list, click "Browse..." and locate it.
   - Common locations:
     - `C:\Users\<YourName>\AppData\Local\Programs\Python\Python3X\python.exe`
     - If you use Anaconda: `C:\Users\<YourName>\anaconda3\python.exe`
     - If you use an editor/IDE (VS Code, PyCharm, etc.), pick the interpreter used by your project.
5. Set the preferred GPU:
   - With `python.exe` (or `pythonw.exe`) selected, find "2. Select the preferred graphics processor for this program."
   - Change the setting from "Global setting (Auto-select)" to "High-performance NVIDIA processor."
   - Click "Apply" and rerun your script.
6. Alternative (Windows 10/11 Settings):
   - Go to Settings → System → Display → Graphics.
   - Under "Add an app," select "Desktop app" and browse for your `python.exe`.
   - Click "Options" and set it to "High performance."

## How it works

Using a simple python script, it determines the size of the image, and runs a script on the GPU using GLSL to create that image.
It then saves the image as a .tiff, which has the advantage of not being compressed.

## Why?

I was bored one day, and kinda wanted to jokingly mildly sabotage a friend's project, for which a sufficiently large image should suffice.

A couple of days later, I found myself texting with said friend about how to make this project work, optimise it, and with another friend with better hardware about how he should run it.