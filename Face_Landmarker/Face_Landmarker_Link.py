import mediapipe as mp
import numpy as np
import cv2
import csv
import transforms3d

from mediapipe.python.solutions import face_mesh
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import tkinter as tk
from tkinter import filedialog

import os

#first questions
headbool = True

while True:
    headtracking = input("Enable head tracking? (Y/N): ")

    if headtracking.lower() == 'y':
        print("Head tracking set")
        headbool = True
        break  # Exit the loop when input is 'Y' or 'y'
    elif headtracking.lower() == 'n':
        print("Head tracking disabled. Pitch, Yaw, and Roll will be 0 for all frames.")
        headbool = False
        break  # Exit the loop when input is 'N' or 'n'
    else:
        print("Invalid input. Please enter 'Y' or 'N'.")

eyebool = True

while True:
    eyesymmetry = input("Do you want symmetric eye tracking?  (Y/N):")

    if eyesymmetry.lower() == 'y':
        print("Eye symmetry set. Left eye movement will be applied to both eyes") 
        eyebool = False 
        break  # Exit the loop when input is 'N' or 'n'              
    elif eyesymmetry.lower() == 'n':
        print("Eye symmetry disabled")
        eyebool = True  
        break  # Exit the loop when input is 'N' or 'n'              
    else:
        print("Invalid input. Please enter 'Y' or 'N'.")        


# media pipe face landmarker options
model_path = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the video mode:
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True)


### DEFINE DRAW LANDMARKS
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image

######
from face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)
# define head rotation - taken from (Jim West - MEFAMO)

# points of the face model that will be used for SolvePnP later
points_idx = [1, 33, 263, 61, 291, 199]
#points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

# Calculates the 3d rotation and 3d landmarks from the 2d landmarks
def calculate_rotation(face_landmarks, pcf, image_shape):
    frame_width = image_shape.width
    frame_height = image_shape.height
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeff = np.zeros((4, 1))

    landmarks = np.array(
        [(lm.x, lm.y, lm.z) for lm in face_landmarks.face_landmarks[0][:468]]

    )
    # print(landmarks.shape)
    landmarks = landmarks.T

    metric_landmarks, pose_transform_mat = get_metric_landmarks(
        landmarks.copy(), pcf
    )

    model_points = metric_landmarks[0:3, points_idx].T
    image_points = (
        landmarks[0:2, points_idx].T
        * np.array([frame_width, frame_height])[None, :]
    )

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeff,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    return pose_transform_mat, metric_landmarks, rotation_vector, translation_vector

####### Use OpenCV’s VideoCapture to load the input video.


# here i make an open file dialogue window
# Create a Tkinter root window
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open the file dialog to select the video file
file_path = filedialog.askopenfilename(title="Select Video File")

# Check if the user selected a file
if file_path:
    # Continue with the rest of your code using the file_path variable
    # ...
    video_path = file_path
else:
    print("No file selected.")
    


# Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
# You’ll need it to calculate the timestamp for each frame.

# Path to the input video file
#video_path = "???.mp4"

# Path to the output CSV file
file_name = os.path.basename(file_path)
# print (file_name)
file_result, extension = os.path.splitext(file_path)
# print(file_result)
output_csv_path = str(file_result) + ".csv"
print (output_csv_path)

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get the frame rate of the video
frame_rate = cap.get(cv2.CAP_PROP_FPS)
# frame_rate = 60

# Calculate the timestamp for each frame
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
total_time = frame_count / frame_rate

# print("Frame rate:", frame_rate)
# print("Total time:", total_time, "seconds")

# List of blendshape names
blendshape_names = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft",
    "EyeSquintLeft", "EyeWideLeft", "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight",
    "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight", "EyeWideRight", "JawForward",
    "JawRight", "JawLeft", "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight",
    "MouthLeft", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper", "MouthPressLeft",
    "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight", "MouthUpperUpLeft",
    "MouthUpperUpRight", "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft",
    "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft", "CheekSquintRight", "NoseSneerLeft",
    "NoseSneerRight", "TongueOut", "HeadYaw", "HeadPitch", "HeadRoll", "LeftEyeYaw",
    "LeftEyePitch", "LeftEyeRoll", "RightEyeYaw", "RightEyePitch", "RightEyeRoll"
]
# Create a CSV file and write the header
header = ["Timecode", "BlendShapeCount"] + blendshape_names
with open(output_csv_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    
    # Loop through each frame
    while cap.isOpened():
        # Read the current frame
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Get the current frame index
        frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Calculate the timestamp for the current frame
        frame_timestamp_ms = int(frame_index * (1000 / frame_rate)) 
        milliseconds = int((frame_index) % frame_rate)   # 60 is the Live link default
        seconds = int((frame_timestamp_ms / 1000) % 60)
        minutes = int((frame_timestamp_ms / (1000 * 60)) % 60)
        hours = int(frame_timestamp_ms / (1000 * 60 * 60))

        frame_index_formatted = int(frame_index % 1000)

        time_formatted = f"{(hours):02d}:{minutes:02d}:{seconds:02d}:{milliseconds:02}.{frame_index_formatted:03d}"
        # print("Formatted Time:", time_formatted)

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        frame_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_array)

        

        # Perform face landmarking on the provided single image.
        # The face landmarker must be created with the video mode.
        with FaceLandmarker.create_from_options(options) as landmarker:
            face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            # print(face_landmarker_result)

            if len(face_landmarker_result.face_blendshapes) > 0:
                face_blendshapes = face_landmarker_result.face_blendshapes[0]
            else:
                 # Skip the frame and continue to the next iteration
                continue
            
            # Create a list to hold all blendshape scores
            all_blendshape_scores = []
            left_iris = face_landmarker_result.face_landmarks [0][468].x
            right_iris = face_landmarker_result.face_landmarks [0][473].x


            
            # !!! for all intents and purposes, and also because of noise, 
            # it is more practical to just duplicate left and right eye data
            # wont be able to do cross eyes.
            

            left_iris_x = face_landmarker_result.face_landmarks [0][468].x
            left_iris_y = face_landmarker_result.face_landmarks [0][468].y
            right_iris_x = face_landmarker_result.face_landmarks [0][473].x
            right_iris_y = face_landmarker_result.face_landmarks [0][473].y
            # print ("left iris: " + str(left_iris))
            
            # Iterate through the face blendshapes starting from index 1 to skip the neutral shape
            for face_blendshapes_category in face_blendshapes[1:]:
                blendshape_name = face_blendshapes_category.category_name
                blendshape_score = face_blendshapes_category.score
                formatted_score = "{:.8f}".format(blendshape_score)
                # print(blendshape_name + ":" + formatted_score)

                # Process or store the blendshape name and score                

                # Add the formatted score to the list
                all_blendshape_scores.append(formatted_score)

                 
            #The order of the indexes in this list needs to be remade
            new_order = [8, 10, 12, 14, 16, 18, 20, 9, 11, 13, 15, 17, 19, 21, 22, 25, 23, 24, 26, 31, 37, 38, 32, 43, 44, 29, 30, 27, 28, 45, 46, 39, 40, 41, 42, 35, 36, 33, 34, 47, 48, 0, 1, 2, 3, 4, 5, 6, 7, 49, 50]
            all_blendshape_scores_sorted =[all_blendshape_scores[i] for i in new_order]
            # print("New one: " + str(all_blendshape_scores_sorted))

            num_blendshapes = len(face_blendshapes[1:])  # Exclude the first blendshape (neutral shape)
            # print("Number of found blendshapes:", num_blendshapes)    

            # Tongue
            tongue = [0]

            ####head rotation
            image_width = mp_image.width
            image_height = mp_image.height

             # pseudo camera internals
            focal_length = image_width
            center = (image_width / 2, image_height / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype="double",
                )

            pcf = PCF(
                near=1,
                far=10000,
                frame_height=image_height,
                frame_width=image_width,
                fy=camera_matrix[1, 1],
                )
            
            pose_transform_mat, metric_landmarks, rotation_vector, translation_vector = calculate_rotation (face_landmarker_result, pcf,  mp_image)
            # print (pose_transform_mat, metric_landmarks, rotation_vector, translation_vector)

            # calculate the head rotation out of the pose matrix
            eulerAngles = transforms3d.euler.mat2euler(pose_transform_mat)
            pitch = -eulerAngles[0]
            yaw = eulerAngles[1]
            roll = eulerAngles[2]

            #final head rotation
            if (headbool == True):                    
                headrotation = [pitch, yaw, roll]
            else:
                headrotation = [ 0, 0, 0]
            
        

            #eye rotation
            if (eyebool == True):                    
                eyes = [left_iris_x, left_iris_y , 0, right_iris_x, right_iris_y , 0,]
            else:
                eyes = [left_iris_x, left_iris_y , 0, left_iris_x, left_iris_y , 0,]

            
            blendshape_data = [time_formatted] +  [num_blendshapes] + all_blendshape_scores_sorted + tongue + headrotation + eyes 


        

        # Write the data row to the CSV file
        writer.writerow(blendshape_data)     

        # Display the frame drawing FaceMesh and timestamp
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), face_landmarker_result)        
        # cv2.imshow("Frame", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
        # Define the value you want to display
        value = time_formatted

        # Convert the value to a string
        text = str(value) + " " +file_name      

        # Specify the font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 0, 0)  # Green color (BGR format)
        thickness = 1
        # Position the text on the image
        position = (50, 50)  # Coordinates of the top-left corner of the text
        

        # Draw the text on the overlay image
        image_w_text = cv2.putText(annotated_image, text, position, font, font_scale, color, thickness)        

        cv2.imshow("Frame", cv2.cvtColor(image_w_text, cv2.COLOR_RGB2BGR))
        
        # Display the frame and timestamp
        # cv2.imshow("Frame", frame)
        # print("Frame:", frame_index, "Timestamp:", frame_timestamp_ms, "milliseconds")

        # Check if the 'q' key is pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()


