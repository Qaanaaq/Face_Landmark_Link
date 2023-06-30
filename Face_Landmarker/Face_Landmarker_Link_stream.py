import mediapipe as mp
import numpy as np
import cv2
import csv

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import matplotlib.pyplot as plt

# madiea pipe face landmarker options
model_path = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

stream_result = []
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
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=stream)


### DEFINE DRAW LANDMARKS
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

####### Use OpenCV’s VideoCapture to load the input video.
    


# Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
# You’ll need it to calculate the timestamp for each frame.


# Create a VideoCapture object
cap = cv2.VideoCapture(0)

frame_rate = cap.get(cv2.CAP_PROP_FPS)


# print("framerate", frame_rate)

# Get the current frame index
frame_index = 0




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
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_array)
        
    # Perform face landmarking on the provided single image.
    # The face landmarker must be created with the video mode.
    with FaceLandmarker.create_from_options(options) as landmarker:
        face_landmarker_result = landmarker.detect_async(mp_image, frame_timestamp_ms)
        # show_image(face_landmarker_result, mp.Image, int(frame_timestamp_ms))
        # print(face_landmarker_result)

        # if len(face_landmarker_result.face_blendshapes[0]) > 0:
        #     face_blendshapes = face_landmarker_result.face_blendshapes[0]
        # else:
        #         # Skip the frame and continue to the next iteration
        #     continue
        
        
        # Iterate through the face blendshapes starting from index 1 to skip the neutral shape
        # for face_blendshapes_category in face_blendshapes[1:]:
        #     blendshape_name = face_blendshapes_category.category_name
        #     blendshape_score = face_blendshapes_category.score
        #     formatted_score = "{:.8f}".format(blendshape_score)
        #     # print(blendshape_name + ":" + formatted_score)   

    # Display the frame drawing FaceMesh and timestamp
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), stream_result)        
    cv2.imshow("Frame", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # cv2.imshow("Frame", cv2.cvtColor(mp_image.numpy_view() , cv2.COLOR_RGB2BGR))

      
    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()

