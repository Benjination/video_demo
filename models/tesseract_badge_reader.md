# Tesseract badge reader

## What it is

This project uses Tesseract OCR to read attendee badge text from the received video stream in `-speedClient` mode.

## Library usage

The OCR integration uses the Tesseract C++ API from the `tesseract` package:

- `#include <tesseract/baseapi.h>`
- `#include <tesseract/resultiterator.h>`
- `tesseract::TessBaseAPI`
- `tesseract::PSM_SPARSE_TEXT`
- `tesseract::ResultIterator`

The code creates a worker thread and initializes Tesseract with:

- `api_.Init(NULL, "eng")`
- `api_.SetPageSegMode(tesseract::PSM_SPARSE_TEXT)`

It then runs OCR on grayscale video frames and parses the detected lines into structured badge data.

## Source

This is not a model file in the same sense as the ONNX hand model or the Haar cascade. It is a third-party OCR engine installed from the Ubuntu packages:

- `tesseract-ocr`
- `tesseract-ocr-eng`
- `libtesseract-dev`
- `libleptonica-dev`

The README explicitly mentions these dependencies for badge reading.

## Where it is tied into the program

The badge OCR logic is wired into `video_broadcast.cpp` as follows:

- `BadgeOcrWorker` is created in `client_read_video_loop(...)`
- the worker receives grayscale input frames from the video stream
- OCR results are parsed by `parse_badge_lines(...)`
- recognized badges are saved to `badges/visible/` and eventually archived with the badge store logic

The main pipeline is:

- `ocr_worker->submit(luma)`
- `BadgeOcrWorker::run()` calls `api_.Recognize(...)`
- `parse_badge_lines(lines, &info)` interprets the text
- `save_badge(...)` stores the extracted name/company/location

## Purpose

This component reads printed badge text from the received video, extracts attendee identity fields, and shows them beside the video in the speed client UI. It is separate from the computer-vision overlays for face and hand detection.
