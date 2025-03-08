# Project Title

This project utilizes MediaPipe's Face Landmarker to process video files and extract facial blendshape data, saving the results to a CSV file.

## Modifications

### 1. GitHub Workflow Update

The `.yaml` configuration file has been updated to ensure successful application builds on GitHub.

### 2. Windows User Enhancements

Several issues specific to Windows users have been addressed:

- **Model Path Resolution:** The application now dynamically constructs the path to the `face_landmarker.task` model file, ensuring it's correctly located regardless of the user's environment.

- **File Dialog Behavior:** The file explorer dialog is now configured to appear in the foreground, preventing it from opening behind other applications.

### 3. Output CSV Directory Control

Users can now specify the directory for the generated CSV file, providing greater flexibility in managing output data.

## Building the Executable Locally

To build the executable file locally, follow these steps:

1. **Upgrade pip:**
    ```bash
   python -m pip install --upgrade pip
   ```


2. **Install `flake8`:**
    ```bash
   pip install flake8
   ```


3. **Install Required Packages:**
    ```bash
   pip install -r Face_Landmarker/requirements.txt
   ```


4. **Build the Application:**
   - **Option 1:** Run the script directly:
      ```bash
     python Face_Landmarker/Face_Landmarker_Link.py
     ```  
   - **Option 2:** Use PyInstaller to create an executable:
      ```bash
     pyinstaller Face_Landmarker/Face_Landmarker_Link.spec
     ```  
      This command will generate the executable in the `dist` directory.  

**Note:** Ensure that all dependencies are installed and properly configured before building the application, also you can builded it on github

For more information on creating executables with PyInstaller, refer to the [PyInstaller Usage Guide](https://pyinstaller.org/en/v4.1/usage.html).

By following these instructions, you can successfully build and run the application on your local machine. 