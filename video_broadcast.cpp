#include <SDL2/SDL.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "sisci_api.h"
#include "sisci_error.h"
#include "sisci_types.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

// Macros for simpler error handling
#define SISCI_ERROR_CHECK(func_name, error)                                  \
    if (error != SCI_ERR_OK) {                                               \
        fprintf(stderr, "%s failed - Error code: 0x%x\n", func_name, error); \
        fprintf(stderr, "   %s\n", SCIGetErrorString(error));                \
        return error;                                                        \
    }

#define ERROR_CHECK_NOT_0(func_name, error)                                \
    if (error != 0) {                                                      \
        fprintf(stderr, "%s failed - Error code: %d\n", func_name, error); \
        return error;                                                      \
    }

#define ERROR_CHECK_NEGATIVE(func_name, error)                             \
    if (error < 0) {                                                       \
        fprintf(stderr, "%s failed - Error code: %d\n", func_name, error); \
        return error;                                                      \
    }

#define ERROR_CHECK_NULL(func_name, ptr)                           \
    if (ptr == NULL) {                                             \
        fprintf(stderr, "%s failed - Returned NULL\n", func_name); \
        return 1;                                                  \
    }

// Some standard sisci arguments
#define NO_CALLBACK NULL
#define NO_OFFSET 0
#define NO_FLAGS 0

// Group ids can be 0-1 for PX, and 0-3 for MX
#define DEFAULT_MULTICAST_GROUP_ID 0

#define NO_VIDEO ""
#define MAX_LEN_VIDEO_PATH 1024

// These dimensions are also used to set the segment size for the client,
// max sizes are used to avoid segfaults when the sizes do not match.
// (Segment sizes can be increased in SISCI, but preallocation is quite conservative
//  so by default they are just 16MiB)
#define MAX_WINDOW_WIDTH 1280   
#define MAX_WINDOW_HEIGHT 720

const int NUM_SIZE_PRESETS = 4;
const int DEFAULT_WINDOW_SIZE_PRESET = 2;
const int WINDOW_WIDTH_PRESETS[NUM_SIZE_PRESETS] = { 426, 640, 854, 1280 };
const int WINDOW_HEIGHT_PRESETS[NUM_SIZE_PRESETS] = { 240, 360, 480, 720 };

const char* FACE_CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
const char* PALM_MODEL_PATH = "models/palm_detection_mediapipe_2023feb.onnx";

int adapter_no = 0;
bool print_frame_status = false;

struct InferenceOverlay {
    cv::CascadeClassifier cascade;
    std::vector<cv::Rect> detections;
    uint32_t frame_count;
};

struct HandInferenceOverlay {
    cv::dnn::Net model;
    std::vector<cv::Rect> detections;
    std::vector<cv::Point2f> anchors;
    uint32_t frame_count;
};

static bool initialize_inference(InferenceOverlay* inference, const char* cascade_path, const char* inference_name)
{
    inference->frame_count = 0;
    if (!inference->cascade.load(cascade_path)) {
        fprintf(stderr, "Error: Could not load %s model: %s\n", inference_name, cascade_path);
        return false;
    }
    return true;
}

static void run_inference(InferenceOverlay* inference, AVFrame* frame, int window_width, int window_height)
{
    constexpr uint32_t INFERENCE_INTERVAL_FRAMES = 6;
    if (inference->frame_count++ % INFERENCE_INTERVAL_FRAMES != 0) {
        return;
    }

    cv::Mat yuv_frame(window_height * 3 / 2, window_width, CV_8UC1, frame->data[0], frame->linesize[0]);
    cv::Mat bgr_frame;
    cv::Mat grayscale_frame;
    cv::cvtColor(yuv_frame, bgr_frame, cv::COLOR_YUV2BGR_I420);
    cv::cvtColor(bgr_frame, grayscale_frame, cv::COLOR_BGR2GRAY);
    cv::Mat original_grayscale_frame = grayscale_frame.clone();
    cv::equalizeHist(grayscale_frame, grayscale_frame);
    inference->cascade.detectMultiScale(grayscale_frame, inference->detections, 1.1, 3, 0, cv::Size(24, 24));

    // Fall back to raw grayscale if histogram equalization hurt detection.
    if (inference->detections.empty()) {
        inference->cascade.detectMultiScale(original_grayscale_frame, inference->detections, 1.1, 3, 0, cv::Size(24, 24));
    }
}

static SDL_Rect compute_video_destination_rect(SDL_Renderer* renderer, int frame_width, int frame_height)
{
    int output_width = frame_width;
    int output_height = frame_height;
    if (SDL_GetRendererOutputSize(renderer, &output_width, &output_height) != 0 || output_width <= 0 || output_height <= 0) {
        return SDL_Rect { 0, 0, frame_width, frame_height };
    }

    float width_scale = (float)output_width / frame_width;
    float height_scale = (float)output_height / frame_height;
    float scale = std::min(width_scale, height_scale);
    int scaled_width = (int)std::round(frame_width * scale);
    int scaled_height = (int)std::round(frame_height * scale);
    int offset_x = (output_width - scaled_width) / 2;
    int offset_y = (output_height - scaled_height) / 2;

    return SDL_Rect { offset_x, offset_y, scaled_width, scaled_height };
}

static SDL_Rect map_detection_to_destination(const cv::Rect& detection, int frame_width, int frame_height, const SDL_Rect& destination_rect)
{
    int left = destination_rect.x + (int)std::round((float)detection.x * destination_rect.w / frame_width);
    int top = destination_rect.y + (int)std::round((float)detection.y * destination_rect.h / frame_height);
    int right = destination_rect.x + (int)std::round((float)(detection.x + detection.width) * destination_rect.w / frame_width);
    int bottom = destination_rect.y + (int)std::round((float)(detection.y + detection.height) * destination_rect.h / frame_height);
    return SDL_Rect { left, top, right - left, bottom - top };
}

static void draw_inference_overlay(
    SDL_Renderer* renderer,
    const InferenceOverlay* inference,
    int frame_width,
    int frame_height,
    const SDL_Rect& destination_rect,
    uint8_t red,
    uint8_t green,
    uint8_t blue)
{
    SDL_SetRenderDrawColor(renderer, red, green, blue, SDL_ALPHA_OPAQUE);
    for (const cv::Rect& detection : inference->detections) {
        SDL_Rect rectangle = map_detection_to_destination(detection, frame_width, frame_height, destination_rect);
        SDL_RenderDrawRect(renderer, &rectangle);
    }
}

static bool initialize_hand_inference(HandInferenceOverlay* inference)
{
    inference->frame_count = 0;
    inference->model = cv::dnn::readNet(PALM_MODEL_PATH);
    if (inference->model.empty()) {
        fprintf(stderr, "Error: Could not load hand detection model: %s\n", PALM_MODEL_PATH);
        return false;
    }

    for (int grid_size : { 24, 12 }) {
        int anchors_per_cell = grid_size == 24 ? 2 : 6;
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                for (int anchor = 0; anchor < anchors_per_cell; anchor++) {
                    inference->anchors.emplace_back((x + 0.5F) / grid_size, (y + 0.5F) / grid_size);
                }
            }
        }
    }
    return inference->anchors.size() == 2016;
}

static void run_hand_inference(HandInferenceOverlay* inference, AVFrame* frame, int window_width, int window_height)
{
    constexpr uint32_t INFERENCE_INTERVAL_FRAMES = 10;
    constexpr int MODEL_SIZE = 192;
    if (inference->frame_count++ % INFERENCE_INTERVAL_FRAMES != 0) {
        return;
    }

    cv::Mat yuv_frame(window_height * 3 / 2, window_width, CV_8UC1, frame->data[0], frame->linesize[0]);
    cv::Mat bgr_frame;
    cv::cvtColor(yuv_frame, bgr_frame, cv::COLOR_YUV2BGR_I420);

    float scale = std::min((float)MODEL_SIZE / bgr_frame.cols, (float)MODEL_SIZE / bgr_frame.rows);
    int resized_width = std::round(bgr_frame.cols * scale);
    int resized_height = std::round(bgr_frame.rows * scale);
    int horizontal_padding = (MODEL_SIZE - resized_width) / 2;
    int vertical_padding = (MODEL_SIZE - resized_height) / 2;

    cv::Mat resized_frame;
    cv::resize(bgr_frame, resized_frame, cv::Size(resized_width, resized_height));
    cv::Mat model_image(MODEL_SIZE, MODEL_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    resized_frame.copyTo(model_image(cv::Rect(horizontal_padding, vertical_padding, resized_width, resized_height)));

    cv::Mat rgb_image;
    model_image.convertTo(rgb_image, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(rgb_image, rgb_image, cv::COLOR_BGR2RGB);
    int input_dimensions[] = { 1, MODEL_SIZE, MODEL_SIZE, 3 };
    cv::Mat input(4, input_dimensions, CV_32F, rgb_image.data);
    inference->model.setInput(input);

    std::vector<cv::Mat> outputs;
    inference->model.forward(outputs, inference->model.getUnconnectedOutLayersNames());
    if (outputs.size() != 2) {
        inference->detections.clear();
        return;
    }

    cv::Mat* box_output = outputs[0].total() > outputs[1].total() ? &outputs[0] : &outputs[1];
    cv::Mat* score_output = box_output == &outputs[0] ? &outputs[1] : &outputs[0];
    cv::Mat boxes = box_output->reshape(1, 2016);
    cv::Mat scores = score_output->reshape(1, 2016);
    std::vector<cv::Rect> candidate_boxes;
    std::vector<float> candidate_scores;

    for (int index = 0; index < 2016; index++) {
        float score = 1.0F / (1.0F + std::exp(-scores.at<float>(index, 0)));
        if (score < 0.6F) {
            continue;
        }
        float center_x = boxes.at<float>(index, 0) + inference->anchors[index].x * MODEL_SIZE;
        float center_y = boxes.at<float>(index, 1) + inference->anchors[index].y * MODEL_SIZE;
        float width = boxes.at<float>(index, 2);
        float height = boxes.at<float>(index, 3);
        int left = std::max(0, (int)std::round((center_x - width / 2.0F - horizontal_padding) / scale));
        int top = std::max(0, (int)std::round((center_y - height / 2.0F - vertical_padding) / scale));
        int right = std::min(window_width, (int)std::round((center_x + width / 2.0F - horizontal_padding) / scale));
        int bottom = std::min(window_height, (int)std::round((center_y + height / 2.0F - vertical_padding) / scale));
        if (right > left && bottom > top) {
            candidate_boxes.emplace_back(left, top, right - left, bottom - top);
            candidate_scores.push_back(score);
        }
    }

    std::vector<int> selected_indices;
    cv::dnn::NMSBoxes(candidate_boxes, candidate_scores, 0.6F, 0.3F, selected_indices, 1.0F, 2);
    inference->detections.clear();
    for (int index : selected_indices) {
        inference->detections.push_back(candidate_boxes[index]);
    }
}

static void draw_hand_overlay(
    SDL_Renderer* renderer,
    const HandInferenceOverlay* inference,
    int frame_width,
    int frame_height,
    const SDL_Rect& destination_rect)
{
    SDL_SetRenderDrawColor(renderer, 30, 210, 240, SDL_ALPHA_OPAQUE);
    for (const cv::Rect& detection : inference->detections) {
        SDL_Rect rectangle = map_detection_to_destination(detection, frame_width, frame_height, destination_rect);
        SDL_RenderDrawRect(renderer, &rectangle);
        int center_x = detection.x + detection.width / 2;
        int center_y = detection.y + detection.height / 2;
        cv::Rect center_marker(center_x - 12, center_y - 12, 24, 24);
        SDL_Rect mapped_center_marker = map_detection_to_destination(center_marker, frame_width, frame_height, destination_rect);
        int mapped_center_x = mapped_center_marker.x + mapped_center_marker.w / 2;
        int mapped_center_y = mapped_center_marker.y + mapped_center_marker.h / 2;
        int marker_half_width = std::max(3, mapped_center_marker.w / 2);
        int marker_half_height = std::max(3, mapped_center_marker.h / 2);
        SDL_RenderDrawLine(renderer, mapped_center_x - marker_half_width, mapped_center_y, mapped_center_x + marker_half_width, mapped_center_y);
        SDL_RenderDrawLine(renderer, mapped_center_x, mapped_center_y - marker_half_height, mapped_center_x, mapped_center_y + marker_half_height);
    }
}

static void apply_window_mode(SDL_Window* window, bool fullscreen)
{
    if (!fullscreen) {
        return;
    }

    if (SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN_DESKTOP) != 0) {
        fprintf(stderr, "Warning: Could not enter fullscreen mode: %s\n", SDL_GetError());
        SDL_MaximizeWindow(window);
    }
}

static bool set_fullscreen(SDL_Window* window, bool fullscreen)
{
    Uint32 mode = fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0;
    if (SDL_SetWindowFullscreen(window, mode) != 0) {
        fprintf(stderr, "Warning: Could not %s fullscreen mode: %s\n", fullscreen ? "enter" : "exit", SDL_GetError());
        return false;
    }
    return true;
}

static bool handle_window_events(SDL_Window* window, bool* is_fullscreen)
{
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            fprintf(stderr, "Program exited through GUI\n");
            return true;
        }

        if (event.type == SDL_KEYDOWN) {
            SDL_Keycode key = event.key.keysym.sym;
            if (key == SDLK_q) {
                fprintf(stderr, "Program exited through keyboard\n");
                return true;
            }

            if (key == SDLK_F11 || key == SDLK_f) {
                bool requested_fullscreen = !(*is_fullscreen);
                if (set_fullscreen(window, requested_fullscreen)) {
                    *is_fullscreen = requested_fullscreen;
                }
            } else if (key == SDLK_ESCAPE && *is_fullscreen) {
                if (set_fullscreen(window, false)) {
                    *is_fullscreen = false;
                }
            }
        }
    }

    return false;
}

void server_handle_frame(
    AVFrame* frame,
    AVFrame* scaled_frame,
    size_t buffer_size,
    SDL_Renderer* renderer,
    SDL_Texture* texture,
    struct SwsContext* sws_ctx,
    InferenceOverlay* face_inference,
    int window_width,
    int window_height,
    bool use_dma,
    // When using DMA
    sci_local_segment_t local_segment,
    sci_remote_segment_t remote_segment,
    sci_dma_queue_t dma_queue,
    // When using PIO
    void* local_frame_buffer,
    sci_map_t remote_map)
{
    sci_error_t sisci_error;

    // Scale and convert the frame to YUV420P format
    sws_scale(
        sws_ctx,
        frame->data,
        frame->linesize,
        0,
        frame->height,
        scaled_frame->data,
        scaled_frame->linesize);

    run_inference(face_inference, scaled_frame, window_width, window_height);

    // Transfer frame
    if (use_dma) {
        // fprintf(stderr, "Transferring using DMA\n");
        // Transfer frame using DMA
        SCIStartDmaTransfer(
            dma_queue,
            local_segment,
            remote_segment,
            0,
            buffer_size,
            0,
            NO_CALLBACK,
            NULL,
            SCI_FLAG_BROADCAST,
            &sisci_error);
        if (sisci_error != SCI_ERR_OK) {
            fprintf(stderr, "%s failed - Error code: 0x%x\n", "SCIStartDmaTransfer", sisci_error);
            fprintf(stderr, "   %s\n", SCIGetErrorString(sisci_error));
            sleep(1);
        }
    } else {
        // fprintf(stderr, "Transferring using PIO\n");
        // Transfer frame using PIO
        SCIMemCpy(NULL,
            local_frame_buffer,
            remote_map,
            0,
            buffer_size,
            0,
            &sisci_error);
        if (sisci_error != SCI_ERR_OK) {
            fprintf(stderr, "%s failed - Error code: 0x%x\n", "SCIMemCpy", sisci_error);
            fprintf(stderr, "   %s\n", SCIGetErrorString(sisci_error));
        }
    }

    SDL_UpdateYUVTexture(
        texture,
        NULL,
        scaled_frame->data[0],
        scaled_frame->linesize[0],
        scaled_frame->data[1],
        scaled_frame->linesize[1],
        scaled_frame->data[2],
        scaled_frame->linesize[2]);
    SDL_Rect destination_rect = compute_video_destination_rect(renderer, window_width, window_height);
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, &destination_rect);
    draw_inference_overlay(renderer, face_inference, window_width, window_height, destination_rect, 40, 220, 90);
    SDL_RenderPresent(renderer);

    // Wait for DMA queue to be done
    if (use_dma && sisci_error == SCI_ERR_OK) {

        sci_dma_queue_state_t q_state = SCIDMAQueueState(dma_queue);
        if (q_state != SCI_DMAQUEUE_ERROR && q_state != SCI_DMAQUEUE_ABORTED) {

            SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &sisci_error);
            if (sisci_error != SCI_ERR_OK) {
                fprintf(stderr, "%s failed - Error code: 0x%x\n", "SCIStartDmaTransfer", sisci_error);
                fprintf(stderr, "   %s\n", SCIGetErrorString(sisci_error));
            }
        } else {
            fprintf(stderr, "%s returned bad queue state\n", "SCIDMAQueueState");
        }
    }
}

int server_read_video_loop(
    SDL_Window* window,
    AVFormatContext* fmt_ctx,
    int video_stream_index,
    AVCodecContext* codec_ctx,
    AVFrame* frame,
    AVFrame* scaled_frame,
    size_t buffer_size,
    SDL_Renderer* renderer,
    SDL_Texture* texture,
    struct SwsContext* sws_ctx,
    InferenceOverlay* face_inference,
    int window_width,
    int window_height,
    double frame_rate,
    bool live_input,
    bool use_dma,
    bool* is_fullscreen,
    // When using DMA
    sci_local_segment_t local_segment,
    sci_remote_segment_t remote_segment,
    sci_dma_queue_t dma_queue,
    // When using PIO
    void* local_frame_buffer,
    sci_map_t remote_map)
{
    AVPacket packet;

    uint32_t frame_interval_ms = 1e3 / frame_rate;

    uint32_t time_of_last_frame_ms = SDL_GetTicks();
    uint32_t time_since_last_frame_ms;

    while (true) {

        if (av_read_frame(fmt_ctx, &packet) >= 0) {

            if (packet.stream_index == video_stream_index) {
                int ret = avcodec_send_packet(codec_ctx, &packet);
                if (ret < 0) {
                    av_packet_unref(&packet);
                    fprintf(stderr, "Error sending packet to decoder\n");
                    if (live_input) {
                        continue;
                    }
                    return -1;
                }
                while (ret >= 0) {
                    if (handle_window_events(window, is_fullscreen)) {
                        av_packet_unref(&packet);
                        return 0;
                    }

                    ret = avcodec_receive_frame(codec_ctx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    } else if (ret < 0) {
                        av_packet_unref(&packet);
                        fprintf(stderr, "Error receiving frame from decoder\n");
                        return -1;
                    }

                    server_handle_frame(
                        frame,
                        scaled_frame,
                        buffer_size,
                        renderer,
                        texture,
                        sws_ctx,
                        face_inference,
                        window_width,
                        window_height,
                        use_dma,
                        // When using DMA
                        local_segment,
                        remote_segment,
                        dma_queue,
                        // When using PIO
                        local_frame_buffer,
                        remote_map);

                    // Free the frame
                    av_frame_unref(frame);

                    // Control the frame rate of the video by delaying if
                    // we're ahead of schedule
                    time_since_last_frame_ms = SDL_GetTicks() - time_of_last_frame_ms;
                    int32_t delay = time_since_last_frame_ms - frame_interval_ms;
                    time_of_last_frame_ms = SDL_GetTicks();
                    if (delay < 0) {
                        if (print_frame_status) {
                            printf("Frame is %d milliseconds ahead of schedule, sleeping...\n", -delay);
                        }
                        usleep((uint32_t)(-delay) * 1000);
                    } else {
                        if (print_frame_status) {
                            printf("Frame is %d milliseconds behind schedule, continuing immediately\n", delay);
                        }
                    }
                }
            }
            av_packet_unref(&packet);
        } else if (!live_input) {
            av_seek_frame(fmt_ctx, video_stream_index, 0, AVSEEK_FLAG_ANY);
        } else {
            fprintf(stderr, "Error reading camera frame\n");
            return -1;
        }
    }
}

int video_server(
    int window_width,
    int window_height,
    bool fullscreen,
    char* input_path,
    bool camera_input,
    sci_desc_t sd,
    int multicast_group_id,
    bool use_dma)
{
    sci_error_t sisci_error;
    int error;
    bool is_fullscreen = fullscreen;

    printf("Opening %s: %s\n", camera_input ? "camera" : "video", input_path);

    // Open either a media file or a V4L2 camera device.
    AVFormatContext* fmt_ctx = NULL;
    AVDictionary* input_options = NULL;
    if (camera_input) {
        avdevice_register_all();
        av_dict_set(&input_options, "input_format", "mjpeg", 0);
        av_dict_set(&input_options, "video_size", "1280x720", 0);
        av_dict_set(&input_options, "framerate", "30", 0);
    }
    const AVInputFormat* input_format = camera_input ? av_find_input_format("v4l2") : NULL;
    if (camera_input && input_format == NULL) {
        fprintf(stderr, "Error: FFmpeg was built without V4L2 input support\n");
        return -1;
    }
    error = avformat_open_input(&fmt_ctx, input_path, input_format, &input_options);
    av_dict_free(&input_options);
    if (error < 0) {
        fprintf(stderr, "Error: Could not open %s\n", camera_input ? "camera" : "video input");
        fprintf(stderr, "%s failed - Error code: %d\n", "avformat_open_input", error);
        exit(1);
    }
    ERROR_CHECK_NEGATIVE("avformat_open_input", error)
    error = avformat_find_stream_info(fmt_ctx, NULL);
    ERROR_CHECK_NEGATIVE("avformat_find_stream_info", error)

    // Find the video stream
    int video_stream_index = -1;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }

    if (video_stream_index == -1) {
        fprintf(stderr, "Error: Could not find video stream\n");
        return -1;
    }

    // Get the codec and codec context
    const AVCodec* codec = avcodec_find_decoder(fmt_ctx->streams[video_stream_index]->codecpar->codec_id);
    ERROR_CHECK_NULL("avcodec_find_decoder", codec);
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    ERROR_CHECK_NULL("avcodec_alloc_context3", codec_ctx);
    error = avcodec_parameters_to_context(codec_ctx, fmt_ctx->streams[video_stream_index]->codecpar);
    ERROR_CHECK_NEGATIVE("avcodec_parameters_to_context", error);
    error = avcodec_open2(codec_ctx, codec, NULL);
    ERROR_CHECK_NEGATIVE("avcodec_open2", error);

    // Allocate memory for the output frame
    AVFrame* frame = av_frame_alloc();
    ERROR_CHECK_NULL("av_frame_alloc", frame);

    // Create the window and renderer
    Uint32 window_flags = SDL_WINDOW_SHOWN;
    SDL_Window* window = SDL_CreateWindow(
        "Video Broadcast Server",
        0,
        0,
        window_width,
        window_height,
        window_flags);
    ERROR_CHECK_NULL("SDL_CreateWindow", window);
    apply_window_mode(window, fullscreen);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
    ERROR_CHECK_NULL("SDL_CreateRenderer", renderer);
    SDL_Texture* texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_YV12,
        SDL_TEXTUREACCESS_STREAMING,
        window_width,
        window_height);
    ERROR_CHECK_NULL("SDL_CreateTexture", texture);

    InferenceOverlay face_inference;
    if (!initialize_inference(&face_inference, FACE_CASCADE_PATH, "face detection")) {
        return -1;
    }

    // Create the scaler context
    struct SwsContext* sws_ctx = sws_getContext(
        codec_ctx->width,
        codec_ctx->height,
        codec_ctx->pix_fmt,
        window_width,
        window_height,
        AV_PIX_FMT_YUV420P,
        0,
        NULL,
        NULL,
        NULL);
    ERROR_CHECK_NULL("sws_getContext", sws_ctx);

    double frame_rate = av_q2d(fmt_ctx->streams[video_stream_index]->r_frame_rate);
    if (frame_rate <= 0.0) {
        frame_rate = 30.0;
    }
    fprintf(stderr, "frame_rate = %f\n", frame_rate);

    sci_local_segment_t local_segment;
    sci_map_t local_map;
    int buffer_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, window_width, window_height, 1);
    ERROR_CHECK_NEGATIVE("av_image_get_buffer_size", buffer_size);

    /* Create local segment */
    // add multicast group id as unique offset to segment_id=0x123 even though
    // this is not a multicast group in order to avoid collision between local segments on the same node
    SCICreateSegment(sd, &local_segment, 0x123 + multicast_group_id, buffer_size, NO_CALLBACK, NULL, NO_FLAGS, &sisci_error);
    SISCI_ERROR_CHECK("SCICreateSegment", sisci_error);

    /* Prepare the segment */
    SCIPrepareSegment(local_segment, adapter_no, NO_FLAGS, &sisci_error);
    SISCI_ERROR_CHECK("SCIPrepareSegment", sisci_error);

    /* Map local segment to user space */
    void* local_frame_buffer = SCIMapLocalSegment(local_segment, &local_map, 0, buffer_size, NULL, NO_FLAGS, &sisci_error);
    SISCI_ERROR_CHECK("SCIMapLocalSegment", sisci_error);

    // Allocate a buffer for the scaled frame data
    AVFrame* scaled_frame = av_frame_alloc();
    ERROR_CHECK_NULL("av_frame_alloc", scaled_frame);
    error = av_image_fill_arrays(
        scaled_frame->data,
        scaled_frame->linesize,
        (uint8_t*)
            local_frame_buffer,
        AV_PIX_FMT_YUV420P,
        window_width,
        window_height,
        1);
    ERROR_CHECK_NEGATIVE("av_image_fill_arrays", error);

    // Connect to and map map remote segment and set that to be remote_frame_buffer
    sci_remote_segment_t remote_segment;
    sci_map_t remote_map;

    // fprintf(stderr, "Connecting to multicast group with id=%d\n", multicast_group_id);
    SCIConnectSegment(
        sd,
        &remote_segment,
        DIS_BROADCAST_NODEID_GROUP_ALL,
        multicast_group_id,
        adapter_no,
        NO_CALLBACK,
        NULL,
        SCI_INFINITE_TIMEOUT,
        SCI_FLAG_BROADCAST,
        &sisci_error);
    SISCI_ERROR_CHECK("SCIConnectSegment", sisci_error);

    SCIMapRemoteSegment(
        remote_segment,
        &remote_map,
        0,
        buffer_size,
        NULL,
        NO_FLAGS,
        &sisci_error);
    SISCI_ERROR_CHECK("SCIMapRemoteSegment", sisci_error);

    sci_dma_queue_t dma_queue;

    /* Create the single dma queue */
    SCICreateDMAQueue(sd, &dma_queue, adapter_no, 1, NO_FLAGS, &sisci_error);
    SISCI_ERROR_CHECK("SCICreateDMAQueue", sisci_error);

    // start sending video frames to client

    int rc = server_read_video_loop(
        window,
        fmt_ctx,
        video_stream_index,
        codec_ctx,
        frame,
        scaled_frame,
        buffer_size,
        renderer,
        texture,
        sws_ctx,
        &face_inference,
        window_width,
        window_height,
        frame_rate,
        camera_input,
        use_dma,
        &is_fullscreen,
        // When using DMA
        local_segment,
        remote_segment,
        dma_queue,
        // When using PIO
        local_frame_buffer,
        remote_map);

    SCIRemoveDMAQueue(dma_queue, NO_FLAGS, &sisci_error);
    // SISCI_ERROR_CHECK("SCIRemoveDMAQueue", sisci_error);

    SCIUnmapSegment(local_map, NO_FLAGS, &sisci_error);
    // SISCI_ERROR_CHECK("SCIUnmapSegment", sisci_error);
    SCIRemoveSegment(local_segment, NO_FLAGS, &sisci_error);
    // SISCI_ERROR_CHECK("SCIRemoveSegment", sisci_error);

    // Disconnect
    SCIUnmapSegment(remote_map, NO_FLAGS, &sisci_error);
    SCIDisconnectSegment(remote_segment, NO_FLAGS, &sisci_error);

    // Clean up
    avformat_close_input(&fmt_ctx);
    avcodec_free_context(&codec_ctx);
    av_frame_free(&frame);
    sws_freeContext(sws_ctx);

    // Free the scaled frame buffer and AVFrame
    // `scaled_frame->data[0]` points to the same memory as `local_frame_buffer`,
    // so the memory in `local_frame_buffer` is freed here
    // i.e. not necessary to do: av_freep(&scaled_frame->data[0]);
    av_frame_free(&scaled_frame);

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

    return rc;
}

int client_read_video_loop(
    SDL_Window* window,
    AVFrame* scaled_frame,
    SDL_Renderer* renderer,
    SDL_Texture* texture,
    HandInferenceOverlay* hand_inference,
    int window_width,
    int window_height,
    double frame_rate,
    bool* is_fullscreen)
{
    uint32_t frame_interval_ms = 1e3 / frame_rate;

    while (true) {

        uint32_t start_time_ms = SDL_GetTicks();

        if (handle_window_events(window, is_fullscreen)) {
            return 0;
        }

        SDL_UpdateYUVTexture(
            texture,
            NULL,
            scaled_frame->data[0],
            scaled_frame->linesize[0],
            scaled_frame->data[1],
            scaled_frame->linesize[1],
            scaled_frame->data[2],
            scaled_frame->linesize[2]);
        SDL_Rect destination_rect = compute_video_destination_rect(renderer, window_width, window_height);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, &destination_rect);
        run_hand_inference(hand_inference, scaled_frame, window_width, window_height);
        draw_hand_overlay(renderer, hand_inference, window_width, window_height, destination_rect);
        SDL_RenderPresent(renderer);

        // Control the frame rate of the video by delaying if
        // we're ahead of schedule
        uint32_t time_since_start_ms = SDL_GetTicks() - start_time_ms;
        int32_t delay = time_since_start_ms - frame_interval_ms;
        if (delay < 0) {
            usleep((uint32_t)(-delay) * 1000);
        }
    }
}

int video_client(int window_width, int window_height, bool fullscreen, sci_desc_t sd, int multicast_group_id)
{
    int rc = 0;
    bool is_fullscreen = fullscreen;

    sci_error_t sisci_error;

    // Create the window and renderer
    Uint32 window_flags = SDL_WINDOW_SHOWN;
    SDL_Window* window = SDL_CreateWindow(
        "Video Broadcast Client",
        0,
        0,
        window_width,
        window_height,
        window_flags);
    ERROR_CHECK_NULL("SDL_CreateWindow", window);
    apply_window_mode(window, fullscreen);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
    ERROR_CHECK_NULL("SDL_CreateRenderer", renderer);
    SDL_Texture* texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_YV12,
        SDL_TEXTUREACCESS_STREAMING,
        window_width,
        window_height);
    ERROR_CHECK_NULL("SDL_CreateTexture", texture);

    HandInferenceOverlay hand_inference;
    if (!initialize_hand_inference(&hand_inference)) {
        return -1;
    }

    // Allocate a buffer for the scaled frame data
    // size_t buffer_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, window_width, window_height, 1);
    size_t buffer_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT, 1);

    // Create local segment and use it as the scaled frame data
    sci_local_segment_t local_segment;
    sci_map_t local_map;
    volatile void* local_frame_buffer;

    // fprintf(stderr, "Creating segment with multicast group with id=%d\n", multicast_group_id);
    SCICreateSegment(sd, &local_segment, multicast_group_id, buffer_size, NO_CALLBACK, NULL, SCI_FLAG_BROADCAST, &sisci_error);
    SISCI_ERROR_CHECK("SCICreateSegment", sisci_error);
    SCIPrepareSegment(local_segment, adapter_no, NO_FLAGS, &sisci_error);
    SISCI_ERROR_CHECK("SCIPrepareSegment", sisci_error);
    local_frame_buffer = SCIMapLocalSegment(local_segment, &local_map, NO_OFFSET, buffer_size, NULL, NO_FLAGS, &sisci_error);
    SISCI_ERROR_CHECK("SCIMapLocalSegment", sisci_error);

    // Create a scaled frame which uses our local segment as a buffer.
    // Data should regularly come into frame buffer if/while there is a server running
    AVFrame* scaled_frame = av_frame_alloc();
    av_image_fill_arrays(scaled_frame->data, scaled_frame->linesize, (uint8_t*)local_frame_buffer, AV_PIX_FMT_YUV420P, window_width, window_height, 1);

    // The frame rate is technically set by the server, here we just set it to
    // be fairly large. If a frame is shown more than one time, that doesn't really
    // matter, as long as we don't lose frames. The frame rate should be higher
    // than it is on the server side if we don't want to lose any frames.
    int frame_rate = 60;

    // Enter loop which shows frames from the mapped memory area at the given frame rate
    rc = client_read_video_loop(window, scaled_frame, renderer, texture, &hand_inference, window_width, window_height, frame_rate, &is_fullscreen);

    // Unmap and remove local segment
    SCIUnmapSegment(local_map, NO_FLAGS, &sisci_error);
    SCIRemoveSegment(local_segment, NO_FLAGS, &sisci_error);

    // Free the scaled frame buffer and AVFrame
    // `scaled_frame->data[0]` points to the same memory as `local_frame_buffer`,
    // so in this case we do not free it
    // av_freep(&scaled_frame->data[0]);
    av_frame_free(&scaled_frame);

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

    return rc;
}

static void print_preset_sizes(void)
{
    printf(" (size presets:");
    for (int i = 0; i < NUM_SIZE_PRESETS; i++) {
        printf("  %d=%dx%d", i, WINDOW_WIDTH_PRESETS[i], WINDOW_HEIGHT_PRESETS[i]);
    }
    printf(")\n");
}

static void print_usage(void)
{
    printf("Usage of video_broadcast\n");
    printf(" video_broadcast -server -video <video_path> [-a <adapter_no>]\n");
    printf(" video_broadcast -server -camera <camera_device> [-a <adapter_no>]\n");
    printf(" video_broadcast -client [-a <adapter_no>]\n");
    printf("\n");
    printf("Arguments:\n");
    printf(" -server / -client              : Whether this program should act as server or client\n");
    printf(" -video <video_path>            : Path to video file (one server input option)\n");
    printf(" -camera <camera_device>        : V4L2 camera device, e.g. /dev/video0 (one server input option)\n");
    printf(" -a <adapter_number>            : Adapter number, (when incorrect SCIPrepareSegment will fail among other things)\n");
    printf(" -mcastgroup <mcast_group_id>   : Multicast group id used to send/receive video (default=%d)\n", DEFAULT_MULTICAST_GROUP_ID);
    printf(" -pio                           : Have server use pio for memory transfer instead of dma\n");
    printf(" -printframestatus              : Have server print out how far ahead or behind the frame is to its schedule.\n");
    printf("                                  Note: Time is mainly spent decoding, scaling, and showing frames, not on sending.\n");
    printf(" -fullscreen                    : Start the display in fullscreen mode\n");
    printf(" Runtime keys                   : F11/f toggle fullscreen, Esc exits fullscreen, q quits\n");
    printf(" -width <window_width>          : Sets the window width in pixels (alternative to size preset, max=%d)\n", MAX_WINDOW_WIDTH);
    printf(" -height <window_height>        : Sets the window height in pixels (alternative to size preset, max=%d)\n", MAX_WINDOW_HEIGHT);
    printf(" -size <preset_size>            : Sets height and width to a preset size (default=%d)\n", DEFAULT_WINDOW_SIZE_PRESET);
    print_preset_sizes();
    printf("\n");
}

int main(int argc, char** argv)
{

    // Server or client, necessary parameter
    int is_server = -1;

    // Sisci resources
    sci_desc_t sd;

    // Error handling
    sci_error_t sisci_error;
    int error;

    // Width and height of window and scaled frame
    int window_width = WINDOW_WIDTH_PRESETS[DEFAULT_WINDOW_SIZE_PRESET];
    int window_height = WINDOW_HEIGHT_PRESETS[DEFAULT_WINDOW_SIZE_PRESET];
    bool fullscreen = false;

    // Server input: a video file or a V4L2 camera device.
    bool input_set = false;
    bool camera_input = false;
    char input_path[MAX_LEN_VIDEO_PATH] = NO_VIDEO;

    // Group id for this video
    int multicast_group_id = DEFAULT_MULTICAST_GROUP_ID;

    // Use DMA vs PIO for frame transfer
    bool use_dma = true;

    /* Get parameters */
    for (uint32_t counter = 1; counter < (uint32_t)argc; counter++) {

        if (strcmp("-help", argv[counter]) == 0) {
            print_usage();
            return 0;
        }

        if (strcmp("-client", argv[counter]) == 0) {
            is_server = 0;
            continue;
        }

        if (strcmp("-server", argv[counter]) == 0) {
            is_server = 1;
            continue;
        }

        if (strcmp("-a", argv[counter]) == 0) {
            adapter_no = strtol(argv[counter + 1], (char**)NULL, 10);
            continue;
        }

        if (strcmp("--video-file-path", argv[counter]) == 0 || strcmp("-video", argv[counter]) == 0) {
            strncpy(input_path, argv[counter + 1], MAX_LEN_VIDEO_PATH - 1);
            input_path[MAX_LEN_VIDEO_PATH - 1] = '\0';
            input_set = true;
            camera_input = false;
            continue;
        }

        if (strcmp("-camera", argv[counter]) == 0) {
            strncpy(input_path, argv[counter + 1], MAX_LEN_VIDEO_PATH - 1);
            input_path[MAX_LEN_VIDEO_PATH - 1] = '\0';
            input_set = true;
            camera_input = true;
            continue;
        }

        if (strcmp("-size", argv[counter]) == 0) {
            int window_size_preset = strtol(argv[counter + 1], (char**)NULL, 10);
            if (window_size_preset < 0 || window_size_preset >= NUM_SIZE_PRESETS) {
                fprintf(stderr, "\nError: Invalid size preset, accaptable range is between [0,%d]\n\n", NUM_SIZE_PRESETS - 1);
                return -1;
            }
            window_width = WINDOW_WIDTH_PRESETS[window_size_preset];
            window_height = WINDOW_HEIGHT_PRESETS[window_size_preset];
            continue;
        }

        if (strcmp("-width", argv[counter]) == 0) {
            window_width = strtol(argv[counter + 1], (char**)NULL, 10);
            continue;
        }

        if (strcmp("-height", argv[counter]) == 0) {
            window_height = strtol(argv[counter + 1], (char**)NULL, 10);
            continue;
        }

        if (strcmp("--mcast-group-id", argv[counter]) == 0 || strcmp("-mcastgroup", argv[counter]) == 0) {
            multicast_group_id = strtol(argv[counter + 1], (char**)NULL, 10);
            continue;
        }

        if (strcmp("-pio", argv[counter]) == 0) {
            use_dma = false;
            continue;
        }

        if (strcmp("--print-frame-status", argv[counter]) == 0 || strcmp("-printframestatus", argv[counter]) == 0) {
            print_frame_status = true;
            continue;
        }

        if (strcmp("-fullscreen", argv[counter]) == 0) {
            fullscreen = true;
            continue;
        }
    }

    if (is_server == -1) {
        print_usage();
        return -1;
    }

    if (is_server == 1 && !input_set) {
        fprintf(stderr, "\nError: No server input provided\n\n");
        print_usage();
        return -1;
    }

    if (window_width <= 0 || window_width > MAX_WINDOW_WIDTH) {
        fprintf(stderr, "\nError: Window width of %d pixels is outside acceptable range for this application\n\n", window_width);
        print_usage();
        return -1;
    }

    if (window_height <= 0 || window_height > MAX_WINDOW_HEIGHT) {
        fprintf(stderr, "\nError: Window height of %d pixels is outside acceptable range for this application\n\n", window_height);
        print_usage();
        return -1;
    }

    // Initialize SISCI
    SCIInitialize(NO_FLAGS, &sisci_error);
    SISCI_ERROR_CHECK("SCIInitialize", sisci_error);

    // Open a sisci descriptor
    SCIOpen(&sd, NO_FLAGS, &sisci_error);
    SISCI_ERROR_CHECK("SCIOpen", sisci_error);

    // Initialize SDL
    struct sigaction action;
    sigaction(SIGINT, NULL, &action);
    error = SDL_Init(SDL_INIT_VIDEO);
    ERROR_CHECK_NOT_0("SDL_Init", error);
    // stop SDL from using sigint for other things than exiting the program
    sigaction(SIGINT, &action, NULL);

    int frame_buffer_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, window_width, window_height, 1);
    ERROR_CHECK_NEGATIVE("av_image_get_buffer_size", frame_buffer_size);
    fprintf(stderr, "frame_buffer_size = %d\n", frame_buffer_size);

    if (is_server) {
        video_server(window_width, window_height, fullscreen, input_path, camera_input, sd, multicast_group_id, use_dma);
    } else {
        video_client(window_width, window_height, fullscreen, sd, multicast_group_id);
    }

    // Close sisci descriptor
    SCIClose(sd, NO_FLAGS, &sisci_error);
    SISCI_ERROR_CHECK("SCIClose", sisci_error);

    SDL_Quit();

    SCITerminate();
}
