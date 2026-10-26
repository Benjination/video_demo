# Palm detection model

## Model

`palm_detection_mediapipe_2023feb.onnx`

## Library usage

This file is consumed by OpenCV DNN, not by a custom hand-detection library. In the code, it is loaded with:

- `cv::dnn::readNet(PALM_MODEL_PATH)`
- `inference->model.setInput(input)`
- `inference->model.forward(...)`
- `cv::dnn::NMSBoxes(...)` for final filtering

The project depends on OpenCV development libraries for this path (`libopencv-dev` / OpenCV DNN support).

## Source

This model comes from the OpenCV Zoo MediaPipe hand detection collection and is described in the project README as the Apache-2.0 licensed OpenCV Zoo hand model collection.

It is the model explicitly referenced in the project as:

- `models/palm_detection_mediapipe_2023feb.onnx`

## Where it is tied into the program

The model path is defined near the top of `video_broadcast.cpp`:

- `const char* PALM_MODEL_PATH = "models/palm_detection_mediapipe_2023feb.onnx";`

The actual integration is in the application flow:

- `initialize_hand_inference(...)` loads the .onnx file
- `run_hand_inference(...)` converts each video frame to the model input format, runs inference, and converts the model output into bounding boxes
- `draw_hand_overlay(...)` renders those boxes onto the video display
- `client_read_video_loop(...)` calls `run_hand_inference(hand_inference, scaled_frame, ...)`

## Purpose

This model is used for detecting hands and drawing cyan boxes around detected palms in the client display. It is separate from the server-side face detection subsystem and is not used for face detection.
