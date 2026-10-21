#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <algorithm>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
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
#define MAX_WINDOW_WIDTH 1920
#define MAX_WINDOW_HEIGHT 1080

const int NUM_SIZE_PRESETS = 5;
const int DEFAULT_WINDOW_SIZE_PRESET = 2;
const int WINDOW_WIDTH_PRESETS[NUM_SIZE_PRESETS] = { 426, 640, 854, 1280, 1920 };
const int WINDOW_HEIGHT_PRESETS[NUM_SIZE_PRESETS] = { 240, 360, 480, 720, 1080 };

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

struct SpeedClientMetrics {
    double current_gbps = 0.0;
    double current_latency_ms = 0.0;
    double smoothed_gbps = 0.0;
    double smoothed_latency_ms = 0.0;
    double max_gbps = 32.0;
    uint64_t total_bytes_processed = 0;
    uint64_t frames_processed = 0;
    uint32_t last_ui_update_ms = 0;
    uint32_t sample_start_ms = 0;
    uint64_t sample_bytes = 0;
    uint32_t sample_frames = 0;
    double sample_latency_ms_total = 0.0;
};

struct BadgeRecord {
    std::string filename;
    std::string payload;
    std::string scanned_at;
};

struct BadgeStore {
    std::filesystem::path visible_directory = "badges/visible";
    std::filesystem::path stored_directory = "badges/stored";
    std::vector<BadgeRecord> visible_records;
    std::set<std::string> known_payloads;
};

struct SpeedClientLayout {
    SDL_Rect video;
    SDL_Rect badge_panel;
    SDL_Rect clear_button;
};

static void update_speed_metrics(SpeedClientMetrics* metrics, double gpbs, double latency_ms, uint32_t now_ms)
{
    const double smoothing = 0.18;
    if (metrics->frames_processed == 0) {
        metrics->smoothed_gbps = gpbs;
        metrics->smoothed_latency_ms = latency_ms;
    } else {
        metrics->smoothed_gbps = metrics->smoothed_gbps * (1.0 - smoothing) + gpbs * smoothing;
        metrics->smoothed_latency_ms = metrics->smoothed_latency_ms * (1.0 - smoothing) + latency_ms * smoothing;
    }
    metrics->current_gbps = gpbs;
    metrics->current_latency_ms = latency_ms;
    metrics->frames_processed += 1;
    metrics->max_gbps = std::max(metrics->max_gbps, gpbs);
    metrics->last_ui_update_ms = now_ms;
}

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

static void draw_thick_rect(SDL_Renderer* renderer, const SDL_Rect& rect, uint8_t red, uint8_t green, uint8_t blue)
{
    const int thickness = 8;
    const int outline = 3;
    Uint8 old_r, old_g, old_b, old_a;
    SDL_GetRenderDrawColor(renderer, &old_r, &old_g, &old_b, &old_a);
    // Dark outline on both sides so the box stays visible on any background.
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    for (int i = -outline; i < 0; ++i) {
        SDL_Rect r { rect.x + i, rect.y + i, rect.w - 2 * i, rect.h - 2 * i };
        SDL_RenderDrawRect(renderer, &r);
    }
    for (int i = thickness; i < thickness + outline; ++i) {
        SDL_Rect r { rect.x + i, rect.y + i, rect.w - 2 * i, rect.h - 2 * i };
        if (r.w > 0 && r.h > 0) SDL_RenderDrawRect(renderer, &r);
    }
    SDL_SetRenderDrawColor(renderer, red, green, blue, SDL_ALPHA_OPAQUE);
    for (int i = 0; i < thickness; ++i) {
        SDL_Rect r { rect.x + i, rect.y + i, rect.w - 2 * i, rect.h - 2 * i };
        if (r.w > 0 && r.h > 0) SDL_RenderDrawRect(renderer, &r);
    }
    SDL_SetRenderDrawColor(renderer, old_r, old_g, old_b, old_a);
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
        draw_thick_rect(renderer, rectangle, red, green, blue);
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
        draw_thick_rect(renderer, rectangle, 30, 210, 240);
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

static TTF_Font* open_default_font(int size)
{
    static const char* font_candidates[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
    };

    for (const char* candidate : font_candidates) {
        TTF_Font* font = TTF_OpenFont(candidate, size);
        if (font != NULL) {
            return font;
        }
    }
    return NULL;
}

static void render_text(SDL_Renderer* renderer, TTF_Font* font, int x, int y, const std::string& text, SDL_Color color)
{
    if (font == NULL || renderer == NULL) {
        return;
    }

    SDL_Surface* surface = TTF_RenderText_Blended(font, text.c_str(), color);
    if (surface == NULL) {
        return;
    }

    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (texture != NULL) {
        SDL_Rect dst_rect = { x, y, surface->w, surface->h };
        SDL_RenderCopy(renderer, texture, NULL, &dst_rect);
        SDL_DestroyTexture(texture);
    }
    SDL_FreeSurface(surface);
}

static std::string json_escape(const std::string& value)
{
    std::ostringstream escaped;
    for (unsigned char character : value) {
        switch (character) {
        case '"':
            escaped << "\\\"";
            break;
        case '\\':
            escaped << "\\\\";
            break;
        case '\b':
            escaped << "\\b";
            break;
        case '\f':
            escaped << "\\f";
            break;
        case '\n':
            escaped << "\\n";
            break;
        case '\r':
            escaped << "\\r";
            break;
        case '\t':
            escaped << "\\t";
            break;
        default:
            if (character < 0x20) {
                escaped << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                        << static_cast<int>(character) << std::dec;
            } else {
                escaped << static_cast<char>(character);
            }
        }
    }
    return escaped.str();
}

static bool read_json_string(const std::string& json, const std::string& key, std::string* value)
{
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos || (position = json.find(':', position + marker.size())) == std::string::npos) {
        return false;
    }
    position = json.find('"', position + 1);
    if (position == std::string::npos) {
        return false;
    }

    value->clear();
    for (++position; position < json.size(); ++position) {
        char character = json[position];
        if (character == '"') {
            return true;
        }
        if (character != '\\') {
            value->push_back(character);
            continue;
        }
        if (++position >= json.size()) {
            return false;
        }
        switch (json[position]) {
        case '"': value->push_back('"'); break;
        case '\\': value->push_back('\\'); break;
        case '/': value->push_back('/'); break;
        case 'b': value->push_back('\b'); break;
        case 'f': value->push_back('\f'); break;
        case 'n': value->push_back('\n'); break;
        case 'r': value->push_back('\r'); break;
        case 't': value->push_back('\t'); break;
        case 'u': {
            if (position + 4 >= json.size() || json.compare(position + 1, 2, "00") != 0) {
                return false;
            }
            char high = json[position + 3];
            char low = json[position + 4];
            auto hex_value = [](char digit) -> int {
                if (digit >= '0' && digit <= '9') return digit - '0';
                if (digit >= 'a' && digit <= 'f') return digit - 'a' + 10;
                if (digit >= 'A' && digit <= 'F') return digit - 'A' + 10;
                return -1;
            };
            int high_value = hex_value(high);
            int low_value = hex_value(low);
            if (high_value < 0 || low_value < 0) {
                return false;
            }
            value->push_back(static_cast<char>((high_value << 4) | low_value));
            position += 4;
            break;
        }
        default:
            return false;
        }
    }
    return false;
}

static bool load_badge_directory(
    const std::filesystem::path& directory,
    std::vector<BadgeRecord>* records,
    std::set<std::string>* known_payloads)
{
    std::error_code error;
    if (!std::filesystem::exists(directory, error)) {
        if (error) {
            fprintf(stderr, "Could not inspect badge directory %s: %s\n", directory.c_str(), error.message().c_str());
            return false;
        }
        return true;
    }

    std::filesystem::directory_iterator entry(directory, error);
    if (error) {
        fprintf(stderr, "Could not read badge directory %s: %s\n", directory.c_str(), error.message().c_str());
        return false;
    }
    for (const auto& file : entry) {
        if (!file.is_regular_file(error) || error || file.path().extension() != ".json") {
            error.clear();
            continue;
        }
        std::ifstream input(file.path());
        if (!input) {
            fprintf(stderr, "Could not read badge record %s\n", file.path().c_str());
            continue;
        }
        std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        BadgeRecord record;
        record.filename = file.path().filename().string();
        if (!read_json_string(json, "payload", &record.payload) || record.payload.empty()) {
            fprintf(stderr, "Ignoring invalid badge JSON %s\n", file.path().c_str());
            continue;
        }
        read_json_string(json, "scanned_at", &record.scanned_at);
        known_payloads->insert(record.payload);
        if (records != NULL) {
            records->push_back(record);
        }
    }
    return true;
}

static bool initialize_badge_store(BadgeStore* store)
{
    std::error_code error;
    std::filesystem::create_directories(store->visible_directory, error);
    if (error) {
        fprintf(stderr, "Could not create %s: %s\n", store->visible_directory.c_str(), error.message().c_str());
        return false;
    }
    std::filesystem::create_directories(store->stored_directory, error);
    if (error) {
        fprintf(stderr, "Could not create %s: %s\n", store->stored_directory.c_str(), error.message().c_str());
        return false;
    }
    if (!load_badge_directory(store->visible_directory, &store->visible_records, &store->known_payloads) ||
        !load_badge_directory(store->stored_directory, NULL, &store->known_payloads)) {
        return false;
    }
    std::sort(store->visible_records.begin(), store->visible_records.end(),
        [](const BadgeRecord& left, const BadgeRecord& right) { return left.filename < right.filename; });
    return true;
}

static std::string current_timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time;
    localtime_r(&time, &local_time);
    std::ostringstream timestamp;
    timestamp << std::put_time(&local_time, "%Y-%m-%dT%H:%M:%S")
              << '.' << std::setw(3) << std::setfill('0') << milliseconds.count();
    return timestamp.str();
}

static bool save_badge(BadgeStore* store, const std::string& payload)
{
    if (payload.empty() || store->known_payloads.count(payload) != 0) {
        return true;
    }
    const std::string scanned_at = current_timestamp();
    const auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::string filename = "badge_" + std::to_string(epoch_ms) + ".json";
    const std::filesystem::path path = store->visible_directory / filename;
    std::ofstream output(path);
    if (!output) {
        fprintf(stderr, "Could not create badge record %s\n", path.c_str());
        return false;
    }
    output << "{\n  \"scanned_at\": \"" << json_escape(scanned_at)
           << "\",\n  \"payload\": \"" << json_escape(payload) << "\"\n}\n";
    output.close();
    if (!output) {
        fprintf(stderr, "Could not finish writing badge record %s\n", path.c_str());
        return false;
    }
    store->visible_records.push_back({ filename, payload, scanned_at });
    store->known_payloads.insert(payload);
    fprintf(stderr, "Saved badge QR payload to %s\n", path.c_str());
    return true;
}

static void archive_visible_badges(BadgeStore* store)
{
    std::vector<BadgeRecord> remaining;
    for (const BadgeRecord& record : store->visible_records) {
        const std::filesystem::path source = store->visible_directory / record.filename;
        const std::filesystem::path destination = store->stored_directory / record.filename;
        std::error_code error;
        std::filesystem::rename(source, destination, error);
        if (error) {
            fprintf(stderr, "Could not archive badge record %s: %s\n", source.c_str(), error.message().c_str());
            remaining.push_back(record);
        }
    }
    store->visible_records.swap(remaining);
}

static SpeedClientLayout compute_speed_client_layout(int output_width, int output_height, int frame_width, int frame_height)
{
    const int margin = 18;
    const int gap = 22;
    const int panel_width = std::min(std::max(250, output_width / 3), std::max(250, output_width - 2 * margin));
    const int panel_x = output_width - margin - panel_width;
    const int video_area_width = std::max(1, panel_x - gap - margin);
    const int video_area_height = std::max(1, output_height - 180);
    const float scale = std::min(
        static_cast<float>(video_area_width) / frame_width,
        static_cast<float>(video_area_height) / frame_height);
    const int video_width = std::max(1, static_cast<int>(std::round(frame_width * scale)));
    const int video_height = std::max(1, static_cast<int>(std::round(frame_height * scale)));
    SpeedClientLayout layout;
    layout.video = { margin + (video_area_width - video_width) / 2, margin, video_width, video_height };
    layout.badge_panel = { panel_x, margin, panel_width, std::max(1, output_height - 2 * margin) };
    layout.clear_button = { panel_x + 12, output_height - margin - 54, panel_width - 24, 42 };
    return layout;
}

static int render_wrapped_text(
    SDL_Renderer* renderer,
    TTF_Font* font,
    int x,
    int y,
    int max_width,
    const std::string& text,
    SDL_Color color)
{
    if (font == NULL || text.empty()) {
        return 0;
    }
    int line_height = TTF_FontLineSkip(font);
    std::istringstream words(text);
    std::string word;
    std::string line;
    int lines = 0;
    while (words >> word) {
        std::string candidate = line.empty() ? word : line + " " + word;
        int candidate_width = 0;
        TTF_SizeUTF8(font, candidate.c_str(), &candidate_width, NULL);
        if (!line.empty() && candidate_width > max_width) {
            render_text(renderer, font, x, y + lines * line_height, line, color);
            ++lines;
            line = word;
        } else {
            line = candidate;
        }
    }
    if (!line.empty()) {
        render_text(renderer, font, x, y + lines * line_height, line, color);
        ++lines;
    }
    return lines * line_height;
}

static SDL_Rect draw_badge_panel(
    SDL_Renderer* renderer,
    TTF_Font* font,
    const BadgeStore& store,
    const SpeedClientLayout& layout,
    size_t scroll_offset)
{
    SDL_SetRenderDrawColor(renderer, 24, 28, 36, 255);
    SDL_RenderFillRect(renderer, &layout.badge_panel);
    SDL_SetRenderDrawColor(renderer, 110, 125, 150, 255);
    SDL_RenderDrawRect(renderer, &layout.badge_panel);

    const int left = layout.badge_panel.x + 14;
    const int content_width = layout.badge_panel.w - 28;
    int y = layout.badge_panel.y + 14;
    render_text(renderer, font, left, y,
        "Visible badges (" + std::to_string(store.visible_records.size()) + ")",
        { 255, 255, 255, 255 });
    y += font != NULL ? TTF_FontLineSkip(font) + 10 : 30;

    SDL_Rect list_clip = {
        layout.badge_panel.x + 1,
        y,
        layout.badge_panel.w - 2,
        std::max(0, layout.clear_button.y - y - 8)
    };
    SDL_RenderSetClipRect(renderer, &list_clip);
    for (size_t offset = scroll_offset; offset < store.visible_records.size() && y < list_clip.y + list_clip.h; ++offset) {
        const BadgeRecord& record = store.visible_records[store.visible_records.size() - 1 - offset];
        SDL_SetRenderDrawColor(renderer, 55, 63, 76, 255);
        SDL_Rect card = { left - 6, y, content_width + 12, 4 };
        SDL_RenderFillRect(renderer, &card);
        y += 10;
        int payload_height = 0;
        size_t field_start = 0;
        while (field_start <= record.payload.size()) {
            size_t field_end = record.payload.find('~', field_start);
            const std::string field = record.payload.substr(
                field_start, field_end == std::string::npos ? std::string::npos : field_end - field_start);
            payload_height += render_wrapped_text(
                renderer, font, left, y + payload_height, content_width, field, { 220, 230, 245, 255 });
            if (field_end == std::string::npos) {
                break;
            }
            field_start = field_end + 1;
        }
        y += std::max(payload_height, font != NULL ? TTF_FontLineSkip(font) : 20) + 10;
    }
    SDL_RenderSetClipRect(renderer, NULL);

    SDL_SetRenderDrawColor(renderer, 165, 54, 54, 255);
    SDL_RenderFillRect(renderer, &layout.clear_button);
    SDL_SetRenderDrawColor(renderer, 255, 215, 215, 255);
    SDL_RenderDrawRect(renderer, &layout.clear_button);
    const std::string button_text = "Clear / Archive";
    int text_width = 0;
    int text_height = 0;
    if (font != NULL && TTF_SizeUTF8(font, button_text.c_str(), &text_width, &text_height) == 0) {
        render_text(renderer, font,
            layout.clear_button.x + (layout.clear_button.w - text_width) / 2,
            layout.clear_button.y + (layout.clear_button.h - text_height) / 2,
            button_text, { 255, 255, 255, 255 });
    }
    return layout.clear_button;
}

static void scan_badge_qr(
    cv::QRCodeDetector* detector,
    BadgeStore* store,
    AVFrame* frame,
    int frame_width,
    int frame_height)
{
    cv::Mat grayscale(frame_height, frame_width, CV_8UC1, frame->data[0], frame->linesize[0]);
    std::string payload = detector->detectAndDecode(grayscale);
    if (!payload.empty()) {
        save_badge(store, payload);
    }
}

static void draw_speed_gauge(
    SDL_Renderer* renderer,
    TTF_Font* font,
    const SpeedClientMetrics& metrics,
    const SDL_Rect& video_rect)
{
    const int margin = 12;
    const int panel_w = std::min(360, std::max(220, video_rect.w));
    const int panel_h = 110;
    const int panel_x = video_rect.x;
    const int panel_y = video_rect.y + video_rect.h + margin;
    const int gauge_x = panel_x + 18;
    const int gauge_y = panel_y + 58;
    const int gauge_w = panel_w - 36;
    const int gauge_h = 18;

    int outer_x = video_rect.x - 10;
    int outer_y = video_rect.y - 10;
    int outer_w = std::max(video_rect.w, panel_w) + 20;
    int outer_h = video_rect.h + panel_h + margin + 20;

    SDL_SetRenderDrawColor(renderer, 140, 160, 200, 255);
    SDL_Rect outer = { outer_x, outer_y, outer_w, outer_h };
    SDL_RenderDrawRect(renderer, &outer);

    SDL_SetRenderDrawColor(renderer, 28, 28, 34, 210);
    SDL_Rect panel = { panel_x, panel_y, panel_w, panel_h };
    SDL_RenderFillRect(renderer, &panel);
    SDL_SetRenderDrawColor(renderer, 126, 136, 170, 255);
    SDL_RenderDrawRect(renderer, &panel);

    SDL_SetRenderDrawColor(renderer, 40, 40, 48, 255);
    SDL_Rect gauge_bg = { gauge_x, gauge_y, gauge_w, gauge_h };
    SDL_RenderFillRect(renderer, &gauge_bg);

    double fill_ratio = std::min(1.0, std::max(0.0, metrics.smoothed_gbps / std::max(0.01, metrics.max_gbps)));
    int fill_width = (int)std::round(gauge_w * fill_ratio);
    if (fill_width > 0) {
        SDL_SetRenderDrawColor(renderer, 36, 210, 125, 255);
        SDL_Rect gauge_fill = { gauge_x, gauge_y, fill_width, gauge_h };
        SDL_RenderFillRect(renderer, &gauge_fill);
    }

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_Rect border = { gauge_x, gauge_y, gauge_w, gauge_h };
    SDL_RenderDrawRect(renderer, &border);

    char buffer[128];
    snprintf(buffer, sizeof(buffer), "Multicast: %.3f GB/s", metrics.smoothed_gbps);
    render_text(renderer, font, panel_x + 16, panel_y + 16, buffer, { 255, 255, 255, 255 });
    snprintf(buffer, sizeof(buffer), "Gen4 target: %.0f GB/s", metrics.max_gbps);
    render_text(renderer, font, panel_x + 16, panel_y + 38, buffer, { 220, 240, 255, 255 });
    double utilization_pct = metrics.max_gbps > 0.0 ? (metrics.smoothed_gbps / metrics.max_gbps) * 100.0 : 0.0;
    snprintf(buffer, sizeof(buffer), "Utilization: %.1f%%", utilization_pct);
    render_text(renderer, font, panel_x + 16, panel_y + 82, buffer, { 180, 220, 255, 255 });
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

static bool handle_window_events(
    SDL_Window* window,
    bool* is_fullscreen,
    const SDL_Rect* clear_button = NULL,
    bool* clear_requested = NULL,
    size_t* badge_scroll = NULL,
    size_t badge_count = 0)
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

            if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT &&
                clear_button != NULL && clear_requested != NULL) {
                const int x = event.button.x;
                const int y = event.button.y;
                if (x >= clear_button->x && x < clear_button->x + clear_button->w &&
                    y >= clear_button->y && y < clear_button->y + clear_button->h) {
                    *clear_requested = true;
                }
            }

            if (event.type == SDL_MOUSEWHEEL && badge_scroll != NULL) {
                if (event.wheel.y > 0 && *badge_scroll > 0) {
                    --(*badge_scroll);
                } else if (event.wheel.y < 0 && *badge_scroll + 1 < badge_count) {
                    ++(*badge_scroll);
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
        av_dict_set(&input_options, "video_size", "1920x1080", 0);
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
    fprintf(stderr, "Input video: %dx%d; multicast output: %dx%d\n",
        codec_ctx->width, codec_ctx->height, window_width, window_height);

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
    bool* is_fullscreen,
    bool speed_client,
    TTF_Font* speed_font)
{
    uint32_t frame_interval_ms = 1e3 / frame_rate;
    SpeedClientMetrics metrics;
    BadgeStore badge_store;
    cv::QRCodeDetector qr_detector;
    if (speed_client && !initialize_badge_store(&badge_store)) {
        return 1;
    }
    uint32_t last_qr_scan_ms = 0;
    size_t badge_scroll = 0;

    while (true) {

        uint32_t start_time_ms = SDL_GetTicks();
        auto frame_start = std::chrono::steady_clock::now();
        SpeedClientLayout layout = {};
        SDL_Rect clear_button = {};
        if (speed_client) {
            int output_width = window_width;
            int output_height = window_height;
            SDL_GetRendererOutputSize(renderer, &output_width, &output_height);
            layout = compute_speed_client_layout(output_width, output_height, window_width, window_height);
            clear_button = layout.clear_button;
        }
        bool clear_requested = false;

        if (handle_window_events(
                window,
                is_fullscreen,
                speed_client ? &clear_button : NULL,
                speed_client ? &clear_requested : NULL,
                speed_client ? &badge_scroll : NULL,
                badge_store.visible_records.size())) {
            return 0;
        }
        if (clear_requested) {
            archive_visible_badges(&badge_store);
            badge_scroll = 0;
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

        SDL_Rect destination_rect;
        if (speed_client) {
            destination_rect = layout.video;
        } else {
            destination_rect = compute_video_destination_rect(renderer, window_width, window_height);
        }

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, &destination_rect);

        run_hand_inference(hand_inference, scaled_frame, window_width, window_height);
        auto inference_end = std::chrono::steady_clock::now();
        draw_hand_overlay(renderer, hand_inference, window_width, window_height, destination_rect);

        if (speed_client) {
            uint32_t now_ms = SDL_GetTicks();
            if (last_qr_scan_ms == 0 || now_ms - last_qr_scan_ms >= 300) {
                scan_badge_qr(&qr_detector, &badge_store, scaled_frame, window_width, window_height);
                last_qr_scan_ms = now_ms;
                badge_scroll = badge_store.visible_records.empty()
                    ? 0
                    : std::min(badge_scroll, badge_store.visible_records.size() - 1);
            }
            uint64_t frame_bytes = static_cast<uint64_t>(window_width) * static_cast<uint64_t>(window_height) * 3ULL / 2ULL;
            double measured_latency_ms = std::chrono::duration<double, std::milli>(inference_end - frame_start).count();

            if (metrics.sample_start_ms == 0) {
                metrics.sample_start_ms = now_ms;
            }
            metrics.sample_bytes += frame_bytes;
            metrics.sample_frames += 1;
            metrics.sample_latency_ms_total += measured_latency_ms;
            metrics.total_bytes_processed += frame_bytes;

            if ((now_ms - metrics.last_ui_update_ms) >= 500) {
                double window_elapsed_s = std::max(0.000001, (double)(now_ms - metrics.sample_start_ms) / 1000.0);
                double observed_gbps = (double)metrics.sample_bytes / window_elapsed_s / 1.0e9;
                double observed_latency_ms = metrics.sample_frames > 0 ? metrics.sample_latency_ms_total / metrics.sample_frames : 0.0;
                observed_gbps = std::min(observed_gbps, 32.0);
                update_speed_metrics(&metrics, observed_gbps, observed_latency_ms, now_ms);
                metrics.sample_start_ms = now_ms;
                metrics.sample_bytes = 0;
                metrics.sample_frames = 0;
                metrics.sample_latency_ms_total = 0.0;
            }
            if ((now_ms - metrics.last_ui_update_ms) < 500 || metrics.frames_processed == 0) {
                draw_speed_gauge(renderer, speed_font, metrics, destination_rect);
            }
            draw_badge_panel(renderer, speed_font, badge_store, layout, badge_scroll);
        }

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

int video_client(int window_width, int window_height, bool fullscreen, sci_desc_t sd, int multicast_group_id, bool speed_client)
{
    int rc = 0;
    bool is_fullscreen = fullscreen;

    sci_error_t sisci_error;

    // Create the window and renderer
    Uint32 window_flags = SDL_WINDOW_SHOWN;
    const char* window_title = speed_client ? "Video Broadcast Speed Client" : "Video Broadcast Client";
    SDL_Window* window = SDL_CreateWindow(
        window_title,
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

    TTF_Font* speed_font = speed_client ? open_default_font(18) : NULL;
    if (speed_client && speed_font == NULL) {
        fprintf(stderr, "Warning: Could not open a font for the speed overlay; continuing without text labels\n");
    }

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
    rc = client_read_video_loop(window, scaled_frame, renderer, texture, &hand_inference, window_width, window_height, frame_rate, &is_fullscreen, speed_client, speed_font);

    if (speed_font != NULL) {
        TTF_CloseFont(speed_font);
    }

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
    printf(" video_broadcast -speedClient [-a <adapter_no>]\n");
    printf("\n");
    printf("Arguments:\n");
    printf(" -server / -client / -speedClient : Whether this program should act as server, client, or speed-monitor client\n");
    printf(" -video <video_path>            : Path to video file (one server input option)\n");
    printf(" -camera <camera_device>        : V4L2 camera device, e.g. /dev/video0 (one server input option)\n");
    printf(" -a <adapter_number>            : Adapter number, (when incorrect SCIPrepareSegment will fail among other things)\n");
    printf(" -mcastgroup <mcast_group_id>   : Multicast group id used to send/receive video (default=%d)\n", DEFAULT_MULTICAST_GROUP_ID);
    printf(" -pio                           : Have server use pio for memory transfer instead of dma\n");
    printf(" -printframestatus              : Have server print out how far ahead or behind the frame is to its schedule.\n");
    printf("                                  Note: Time is mainly spent decoding, scaling, and showing frames, not on sending.\n");
    printf(" -fullscreen                    : Start the display in fullscreen mode (default)\n");
    printf(" -windowed                      : Start the display in a window instead of fullscreen\n");
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
    bool fullscreen = true;

    // Server input: a video file or a V4L2 camera device.
    bool input_set = false;
    bool camera_input = false;
    bool speed_client_mode = false;
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

        if (strcmp("-client", argv[counter]) == 0 || strcmp("-speedClient", argv[counter]) == 0 || strcmp("-speedclient", argv[counter]) == 0 || strcmp("-speed-client", argv[counter]) == 0) {
            is_server = 0;
            if (strcmp("-speedClient", argv[counter]) == 0 || strcmp("-speedclient", argv[counter]) == 0 || strcmp("-speed-client", argv[counter]) == 0) {
                speed_client_mode = true;
            }
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

        if (strcmp("-windowed", argv[counter]) == 0) {
            fullscreen = false;
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
    if (TTF_Init() != 0) {
        fprintf(stderr, "Warning: SDL_ttf initialization failed: %s\n", TTF_GetError());
    }
    // stop SDL from using sigint for other things than exiting the program
    sigaction(SIGINT, &action, NULL);

    int frame_buffer_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, window_width, window_height, 1);
    ERROR_CHECK_NEGATIVE("av_image_get_buffer_size", frame_buffer_size);
    fprintf(stderr, "frame_buffer_size = %d\n", frame_buffer_size);

    if (is_server) {
        video_server(window_width, window_height, fullscreen, input_path, camera_input, sd, multicast_group_id, use_dma);
    } else {
        video_client(window_width, window_height, fullscreen, sd, multicast_group_id, speed_client_mode);
    }

    // Close sisci descriptor
    SCIClose(sd, NO_FLAGS, &sisci_error);
    SISCI_ERROR_CHECK("SCIClose", sisci_error);

    TTF_Quit();
    SDL_Quit();

    SCITerminate();
}
