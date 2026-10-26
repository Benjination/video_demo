# Model Notes

This project uses two different inference sources for overlays:

1. Face detection uses an OpenCV Haar cascade XML file provided by the OpenCV package install.
2. Hand detection uses the MediaPipe palm detection ONNX model stored in this folder.

These are independent detectors that are called from different parts of the application, and they do not share a single model pipeline.

## Where they are wired in

The relevant definitions are in `video_broadcast.cpp`:

- `FACE_CASCADE_PATH` points to `/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml`
- `PALM_MODEL_PATH` points to `models/palm_detection_mediapipe_2023feb.onnx`

The face detector is loaded by `initialize_inference(...)` and run by `run_inference(...)`.
The hand detector is loaded by `initialize_hand_inference(...)` and run by `run_hand_inference(...)`.

The client loop calls both in `client_read_video_loop(...)`:

- face overlay: `run_inference(face_inference, scaled_frame, ...)`
- hand overlay: `run_hand_inference(hand_inference, scaled_frame, ...)`

## Library usage

- OpenCV core and DNN APIs are used for both detectors.
- Face detection uses `cv::CascadeClassifier` and `detectMultiScale(...)`.
- Hand detection uses `cv::dnn::readNet(...)`, `cv::dnn::Net`, and `cv::dnn::NMSBoxes(...)`.
- The project also depends on FFmpeg for decoding frames and SDL for rendering overlays.

## Notes

The face detection model is not stored in this repository because it comes from the system OpenCV installation. The palm model is the one included in `models/` and is loaded directly from the local project directory.
