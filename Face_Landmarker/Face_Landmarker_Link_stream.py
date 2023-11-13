import mediapipe as mp
import numpy as np
import cv2
from pylivelinkface import PyLiveLinkFace, FaceBlendShape

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import socket
import time
import random
import transforms3d

# madiea pipe face landmarker options
model_path = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# stream_result = []
stream_result = None  # Initialize as None or with an appropriate default value
# 
# Create a face landmarker instance with the live stream mode:
# def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     print('face landmarker result: {}'.format(result))

def stream (result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print ("updating!")
    global stream_result
    stream_result = result


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),    
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=stream)


### DEFINE DRAW LANDMARKS
from mediapipe.python.solutions.drawing_utils import DrawingSpec
# custom_style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
white_style = DrawingSpec(color=(254, 254, 254), thickness=None, circle_radius = 1) 
yellow_style = DrawingSpec(color=(255, 255, 0), thickness=None, circle_radius = 1) 
none_style = DrawingSpec(color=(0, 0, 0), thickness=None, circle_radius = 0) 


def draw_landmarks_on_image(rgb_image, detection_result):
    # print (detection_result)
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
            .get_default_face_mesh_tesselation_style()
            
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            # connection_drawing_spec=mp.solutions.drawing_styles
            # .get_default_face_mesh_contours_style())
            connection_drawing_spec = white_style)
        
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            # connection_drawing_spec=mp.solutions.drawing_styles
            # .get_default_face_mesh_iris_connections_style())
            connection_drawing_spec=yellow_style)

    return annotated_image

####### Use OpenCV’s VideoCapture to load the input video.
    


# Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
# You’ll need it to calculate the timestamp for each frame.


# Create a VideoCapture object
cap = cv2.VideoCapture(0)

frame_rate = cap.get(cv2.CAP_PROP_FPS)


# print("framerate", frame_rate)

# Get the current frame index
frame_index = 1

#### MEFAMO/pyLiveLinkFace 

UDP_IP = "127.0.0.1"
UDP_PORT = 11111

py_face = PyLiveLinkFace()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
s.connect((UDP_IP, UDP_PORT))

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
        [(lm.x, lm.y, lm.z) for lm in face_landmarks]#.face_landmarks[0][:468]]
    )
    #print(landmarks.shape)
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

# List of blendshape names
blendshape_names = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft",
    "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight",
    "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight", "jawForward",
    "jawRight", "jawLeft", "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthRight",
    "mouthLeft", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft",
    "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft",
    "noseSneerRight", "tongueOut", "headYaw", "headPitch", "headRoll", "leftEyeYaw",
    "leftEyePitch", "leftEyeRoll", "rightEyeYaw", "rightEyePitch", "rightEyeRoll"
]
    
# Loop through each frame
while cap.isOpened():
    # Read the current frame
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        break

    # Increment the frame index
    frame_index += 1
    # print ("Frame index:" + str(frame_index))

    # Calculate the timestamp for the current frame
    frame_timestamp_ms = int(frame_index * (1000 / frame_rate)) 
    # print ("Frame ms:" + str(frame_timestamp_ms))

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    frame_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
     
    width = 600 
    scale_percent = frame.shape[1] / width
    height = int(frame.shape[0] / scale_percent)    
    dim = (width, height)
    # print (frame.shape[0], frame.shape[1],scale_percent, dim)   

    frame_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    frame_array = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_array)
        
    # Perform face landmarking on the provided single image.
    # The face landmarker must be created with the video mode.
    with FaceLandmarker.create_from_options(options) as landmarker:
        face_landmarker_result = landmarker.detect_async(mp_image, frame_timestamp_ms)
        # show_image(face_landmarker_result, mp.Image, int(frame_timestamp_ms))
        # stream(face_landmarker_result, mp_image, frame_timestamp_ms)  # Update stream_result
        
       
        
        
        if stream_result is not None:            
            if len(stream_result.face_blendshapes) > 0:
                face_blendshapes = stream_result.face_blendshapes[0]                
                         
                # Initialize an empty multidimensional array
                blendshape_data = []

                #Iterate through the face blendshapes starting from index 1 to skip the neutral shape
                for i in range(1, min(len(face_blendshapes), 52)):
                    face_blendshapes_category = face_blendshapes[i]
                    blendshape_name = face_blendshapes_category.category_name
                    blendshape_score = face_blendshapes_category.score
                    
                    
                    formatted_score = "{:.3f}".format(blendshape_score)

                    blendshape_data.append([blendshape_name, formatted_score])

                
                # for name, score in zip(blendshape_name, blendshape_score):
                #     blendshape_dict[name] = score
                
                # print (blendshape_data)                
                
                # Initialize an empty array to store the rearranged data
                rearranged_blendshape_data = []

                
                # Iterate through the desired order
                for name in blendshape_names:
                    # Find the corresponding data in the multidimensional array
                    for data in blendshape_data:
                        blendshape_name, formatted_score = data
                        if blendshape_name == name:
                            rearranged_blendshape_data.append([blendshape_name, formatted_score])
                            break  # Stop searching once found              
                    
                    
                  
                for i in range(0, 51):
                    blendshape_name, blendshape_score = rearranged_blendshape_data[i]
                    # print (blendshape_score)
                    blendshape_score = float(blendshape_score) 
                    py_face.set_blendshape(FaceBlendShape(i), blendshape_score, False)                

             

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
                
                if stream_result is not None:
                    if stream_result.face_landmarks:
                        # Face landmarks are detected, use the new values
                        solved_face_landmarks = stream_result.face_landmarks[0][:468]
                        previous_face_landmarks = solved_face_landmarks
                        # print(face_landmarks)
                    else:
                        if previous_face_landmarks is not None:
                            # No face landmarks detected, use the previous values
                            solved_face_landmarks = previous_face_landmarks
                            print("Using previous face landmarks.")
                        else:
                            print("No face landmarks detected and no previous data available.")
                            # Handle the case when no face landmarks are available and no previous data is available
                # print(solved_face_landmarks)


                pose_transform_mat, metric_landmarks, rotation_vector, translation_vector = calculate_rotation (solved_face_landmarks, pcf,  mp_image)
                # print (pose_transform_mat, metric_landmarks, rotation_vector, translation_vector)

                # calculate the head rotation out of the pose matrix
                eulerAngles = transforms3d.euler.mat2euler(pose_transform_mat)
                pitch = -eulerAngles[0]-0.3
                yaw = eulerAngles[1]
                roll = eulerAngles[2]  

                py_face.set_blendshape(FaceBlendShape(51), 0)
                py_face.set_blendshape(FaceBlendShape(52), yaw, False)
                py_face.set_blendshape(FaceBlendShape(53), pitch,False)
                py_face.set_blendshape(FaceBlendShape(54), roll, False)
                # print (py_face.encode())
                s.sendall(py_face.encode())
                

            else:
                print("No face blendshapes detected.")
                # Handle the case when no blendshapes are detected     
        
        else:
            print("stream_result is None")
            # Handle the case when stream_result is None (no face detected)
        
         
    

    # Display the frame drawing FaceMesh and timestamp
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), stream_result)  

    # Convert the value to a string
    text = "Streaming... Hit Q to Exit"    

    # Specify the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    color = (255, 0, 0)  # Red color (BGR format)
    thickness = 1
    # Position the text on the image
    position = (10, 10)  # Coordinates of the top-left corner of the text
    

    # Draw the text on the overlay image
    image_w_text = cv2.putText(annotated_image, text, position, font, font_scale, color, thickness)        

    cv2.imshow("Frame", cv2.cvtColor(image_w_text, cv2.COLOR_RGB2BGR))      
    
    # cv2.imshow("Frame", cv2.cvtColor(mp_image.numpy_view() , cv2.COLOR_RGB2BGR))

      
    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        s.close()
        break


# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()