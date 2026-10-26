# Face detection model

## Model

This project does not keep the face detector in the repository. It loads the OpenCV Haar cascade XML from the installed OpenCV data package:

`/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml`

## Library usage

This is a classic OpenCV Haar cascade detector, loaded through `cv::CascadeClassifier` and run with `detectMultiScale(...)`.

The relevant code in `video_broadcast.cpp` is:

- `const char* FACE_CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";`
- `initialize_inference(...)` loads the classifier
- `run_inference(...)` converts each frame to grayscale and calls `detectMultiScale(...)`

This uses the OpenCV package and the `opencv-data` package on Ubuntu, which provides the Haar cascade XML files.

## Source

The cascade is shipped with OpenCV and installed by the OpenCV data package. It is not a custom model and not a project-local file.

## Where it is tied into the program

The server path loads and uses it before broadcasting frames:

- `initialize_inference(&face_inference, FACE_CASCADE_PATH, "face detection")`
- `run_inference(face_inference, scaled_frame, window_width, window_height)`
- `draw_inference_overlay(renderer, face_inference, ...)`

The client also loads this cascade in `speedClient` mode, where it draws green boxes around detected faces.

## Purpose

This detector is used for face detection and face overlays only. It is separate from the MediaPipe palm detector in `models/palm_detection_mediapipe_2023feb.onnx`.
