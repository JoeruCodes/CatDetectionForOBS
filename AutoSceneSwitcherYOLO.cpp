#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <obs/obs.h>
#include <obs/util/threading.h>

// Set up OBS WebSocket connection
obsws_t *ws = nullptr;

// Load YOLO model and class labels
std::pair<cv::dnn::Net, std::vector<std::string>> load_yolo_model()
{
    cv::dnn::Net net = cv::dnn::readNet("yolov3.weights", "yolov3.cfg");
    std::vector<std::string> classes;
    std::ifstream file("coco.names");
    std::string line;
    while (std::getline(file, line))
    {
        classes.push_back(line);
    }
    return std::make_pair(net, classes);
}

int main()
{
    // Set up OBS WebSocket connection
    obs_data_t *settings = obs_data_create();
    obs_data_set_string(settings, "host", host);
    obs_data_set_int(settings, "port", port);
    obs_data_set_string(settings, "password", password);
    ws = obsws_open(settings);
    obs_data_release(settings);

    // Load YOLO model and class labels
    auto [net, classes] = load_yolo_model();

    // Define scene names and camera indexes
    std::map<std::string, std::variant<int, std::string>> scene_mapping = {
        {"USB Cam", 0},
        {"IP Cam 1", "rtsp://username:password@1.1.1.1"},
        {"IP Cam 2", "rtsp://username:password@1.1.1.1"},
        {"IP Cam 3", "rtsp://username:password@1.1.1.1"},
        {"IP Cam 4", "rtsp://username:password@1.1.1.1"},
        {"IP Cam 5", "rtsp://username:password@1.1.1.1"},
        {"IP Cam 6", "rtsp://username:password@1.1.1.1"}};

    // Open cameras
    std::vector<cv::VideoCapture> cameras;
    for (const auto &entry : scene_mapping)
    {
        if (const auto *index = std::get_if<int>(&entry.second))
        {
            cv::VideoCapture camera(*index);
            cameras.push_back(std::move(camera));
        }
    }

    // Main loop
    int counter = 0; // Frame counter
    while (true)
    {
        std::map<std::string, int> cat_occupancy; // Store cat occupancy for each camera

        // Capture frames from cameras
        for (const auto &entry : scene_mapping)
        {
            const std::string &scene_name = entry.first;
            const auto &camera_index = entry.second;

            if (const auto *index = std::get_if<int>(&camera_index))
            {
                cv::VideoCapture &camera = cameras[*index];
                cv::Mat frame;
                if (camera.read(frame))
                {
                    // Perform object detection on frame
                    cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(416, 416), cv::Scalar(), true, false);
                    net.setInput(blob);
                    std::vector<cv::Mat> layer_outputs;
                    net.forward(layer_outputs, net.getUnconnectedOutLayers());

                    // Initialize variables for cat detection
                    int cat_count = 0;

                    // Loop over each output layer
                    for (const cv::Mat &output : layer_outputs)
                    {
                        for (int i = 0; i < output.rows; i++)
                        {
                            const float *detection = output.ptr<float>(i);
                            float confidence = detection[5];
                            int class_id = std::max_element(detection + 6, detection + output.cols, std::less<float>()) - detection - 6;

                            // Check if the detected object is a cat
                            if (classes[class_id] == "cat" && confidence > 0.5)
                            {
                                cat_count++;
                            }
                        }
                    }

                    // Update cat occupancy dictionary
                    cat_occupancy[scene_name] = cat_count;
                }
            }
        }

        // Find scene with highest cat occupancy every 20 frames
        if (counter == 20)
        {
            std::string max_occupancy_scene = std::max_element(cat_occupancy.begin(), cat_occupancy.end(), [](const auto &a, const auto &b) {
                                                   return a.second < b.second;
                                               })
                                                  ->first;

            // Switch to new scene if different from current scene
            const char *current_scene = obs_source_get_name(obs_get_current_scene());
            if (max_occupancy_scene != current_scene)
            {
                obsws_call(obsws, requests.SetCurrentScene(max_occupancy_scene.c_str()), nullptr, nullptr);
                std::this_thread::sleep_for(std::chrono::seconds(2)); // Delay in seconds between scene switches
            }

            counter = 0; // Reset frame counter
        }
        else
        {
            counter++;
        }

        // Release frames from cameras
        for (cv::VideoCapture &camera : cameras)
        {
            camera.grab();
        }

        // Delay to achieve desired frame rate
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Clean up
    for (cv::VideoCapture &camera : cameras)
    {
        camera.release();
    }
    obsws_close(ws);
    obsws_free(ws);

    return 0;
}
