/**
 * Continuous ACT Policy Inference in C++
 * Uses LibTorch and OpenCV for real-time inference
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace F = torch::nn::functional;

// Normalization statistics structure
struct NormStats {
    std::vector<float> mean;
    std::vector<float> std;
};

// Configuration class
class ACTInferenceConfig {
public:
    std::string checkpoint_dir;
    std::string dataset_dir;
    torch::DeviceType device;
    
    // Model parameters
    int n_action_steps;
    int state_dim;
    int action_dim;
    
    // Normalization stats
    std::map<std::string, NormStats> stats;
    
    ACTInferenceConfig(const std::string& checkpoint, const std::string& dataset, 
                      const std::string& device_str = "cuda") {
        checkpoint_dir = checkpoint;
        dataset_dir = dataset;
        device = (device_str == "cuda" && torch::cuda::is_available()) ? 
                 torch::kCUDA : torch::kCPU;
        
        loadConfig();
        loadStats();
    }
    
private:
    void loadConfig() {
        std::string config_path = checkpoint_dir + "/pretrained_model/config.json";
        std::ifstream config_file(config_path);
        
        if (!config_file.is_open()) {
            std::cerr << "Error: Could not open config file: " << config_path << std::endl;
            throw std::runtime_error("Config file not found");
        }
        
        json config_json;
        try {
            config_file >> config_json;
        } catch (const json::parse_error& e) {
            std::cerr << "Error parsing config file: " << config_path << std::endl;
            std::cerr << "Parse error: " << e.what() << std::endl;
            throw;
        }
        
        n_action_steps = config_json["n_action_steps"];
        state_dim = config_json["input_features"]["observation.state"]["shape"][0];
        action_dim = config_json["output_features"]["action"]["shape"][0];
        
        std::cout << "Loaded config: n_action_steps=" << n_action_steps 
                  << ", state_dim=" << state_dim 
                  << ", action_dim=" << action_dim << std::endl;
    }
    
    void loadStats() {
        std::string stats_path = dataset_dir + "/meta/stats.json";
        std::ifstream stats_file(stats_path);
        
        if (!stats_file.is_open()) {
            std::cerr << "Error: Could not open stats file: " << stats_path << std::endl;
            throw std::runtime_error("Stats file not found");
        }
        
        json stats_json;
        try {
            stats_file >> stats_json;
        } catch (const json::parse_error& e) {
            std::cerr << "Error parsing stats file: " << stats_path << std::endl;
            std::cerr << "Parse error: " << e.what() << std::endl;
            throw;
        }
        
        // Load state stats
        stats["observation.state"] = {
            stats_json["observation.state"]["mean"].get<std::vector<float>>(),
            stats_json["observation.state"]["std"].get<std::vector<float>>()
        };
        
        // Load action stats
        stats["action"] = {
            stats_json["action"]["mean"].get<std::vector<float>>(),
            stats_json["action"]["std"].get<std::vector<float>>()
        };
        
        // Load image stats (simplified - using mean across all pixels)
        std::vector<float> img_main_mean, img_main_std;
        for (auto& channel : stats_json["observation.images.main"]["mean"]) {
            img_main_mean.push_back(channel[0][0]);
        }
        for (auto& channel : stats_json["observation.images.main"]["std"]) {
            img_main_std.push_back(channel[0][0]);
        }
        stats["observation.images.main"] = {img_main_mean, img_main_std};
        
        std::vector<float> img_sec_mean, img_sec_std;
        for (auto& channel : stats_json["observation.images.secondary_0"]["mean"]) {
            img_sec_mean.push_back(channel[0][0]);
        }
        for (auto& channel : stats_json["observation.images.secondary_0"]["std"]) {
            img_sec_std.push_back(channel[0][0]);
        }
        stats["observation.images.secondary_0"] = {img_sec_mean, img_sec_std};
        
        std::cout << "Loaded normalization statistics" << std::endl;
    }
};

// ACT Inference class
class ACTInference {
private:
    ACTInferenceConfig config;
    torch::jit::script::Module model;
    std::deque<double> inference_times;
    const int max_time_samples = 30;
    
public:
    ACTInference(const ACTInferenceConfig& cfg) : config(cfg) {
        loadModel();
    }
    
    void loadModel() {
        std::string model_path = config.checkpoint_dir + "/pretrained_model/model_torchscript.pt";
        
        std::cout << "Loading model from: " << model_path << std::endl;
        
        try {
            model = torch::jit::load(model_path);
            model.to(config.device);
            model.eval();
            std::cout << "Model loaded successfully!" << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            std::cerr << "\nNote: You need to export the PyTorch model to TorchScript first." << std::endl;
            std::cerr << "Run the provided Python script to export the model." << std::endl;
            throw;
        }
    }
    
    torch::Tensor normalize(torch::Tensor data, const std::string& key) {
        if (config.stats.find(key) == config.stats.end()) {
            return data;
        }
        
        const auto& stats = config.stats[key];
        auto mean = torch::from_blob(
            const_cast<float*>(stats.mean.data()),
            {static_cast<long>(stats.mean.size())},
            torch::kFloat32
        ).clone().to(data.device());
        
        auto std = torch::from_blob(
            const_cast<float*>(stats.std.data()),
            {static_cast<long>(stats.std.size())},
            torch::kFloat32
        ).clone().to(data.device());
        
        const auto feature_size = static_cast<int64_t>(mean.numel());
        if (data.dim() >= 1 && mean.dim() == 1) {
            std::vector<int64_t> target_shape(data.dim(), 1);
            bool matched_dim = false;
            for (int64_t dim = 0; dim < data.dim(); ++dim) {
                if (data.size(dim) == feature_size) {
                    target_shape[dim] = feature_size;
                    matched_dim = true;
                    break;
                }
            }
            
            if (matched_dim) {
                mean = mean.view(target_shape);
                std = std.view(target_shape);
            } else if (data.dim() == 1 && data.size(0) == feature_size) {
                // Already aligned for 1D tensors
            } else {
                std::cerr << "Warning: could not automatically align normalization stats for key "
                          << key << ". Falling back to prefix unsqueeze." << std::endl;
                while (mean.dim() < data.dim()) {
                    mean = mean.unsqueeze(0);
                    std = std.unsqueeze(0);
                }
            }
        } else if (mean.dim() != data.dim()) {
            while (mean.dim() < data.dim()) {
                mean = mean.unsqueeze(0);
                std = std.unsqueeze(0);
            }
        }
        
        return (data - mean) / (std + 1e-8);
    }
    
    torch::Tensor unnormalize(torch::Tensor data, const std::string& key) {
        if (config.stats.find(key) == config.stats.end()) {
            return data;
        }
        
        const auto& stats = config.stats[key];
        auto mean = torch::from_blob(
            const_cast<float*>(stats.mean.data()),
            {static_cast<long>(stats.mean.size())},
            torch::kFloat32
        ).clone().to(data.device());
        
        auto std = torch::from_blob(
            const_cast<float*>(stats.std.data()),
            {static_cast<long>(stats.std.size())},
            torch::kFloat32
        ).clone().to(data.device());
        
        const auto feature_size = static_cast<int64_t>(mean.numel());
        if (data.dim() >= 1 && mean.dim() == 1) {
            std::vector<int64_t> target_shape(data.dim(), 1);
            bool matched_dim = false;
            for (int64_t dim = 0; dim < data.dim(); ++dim) {
                if (data.size(dim) == feature_size) {
                    target_shape[dim] = feature_size;
                    matched_dim = true;
                    break;
                }
            }
            
            if (matched_dim) {
                mean = mean.view(target_shape);
                std = std.view(target_shape);
            } else if (data.dim() == 1 && data.size(0) == feature_size) {
                // Already aligned for 1D tensors
            } else {
                std::cerr << "Warning: could not automatically align normalization stats for key "
                          << key << ". Falling back to prefix unsqueeze." << std::endl;
                while (mean.dim() < data.dim()) {
                    mean = mean.unsqueeze(0);
                    std = std.unsqueeze(0);
                }
            }
        } else if (mean.dim() != data.dim()) {
            while (mean.dim() < data.dim()) {
                mean = mean.unsqueeze(0);
                std = std.unsqueeze(0);
            }
        }
        
        return data * std + mean;
    }
    
    torch::Tensor preprocessImage(const cv::Mat& image) {
        // Convert BGR to RGB
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        
        // Convert to float and normalize to [0, 1]
        rgb_image.convertTo(rgb_image, CV_32F, 1.0 / 255.0);
        
        // Convert to tensor (H, W, C) -> (C, H, W)
        torch::Tensor tensor = torch::from_blob(
            rgb_image.data,
            {rgb_image.rows, rgb_image.cols, 3},
            torch::kFloat32
        ).clone();
        
        tensor = tensor.permute({2, 0, 1}); // HWC -> CHW
        
        return tensor;
    }
    
    torch::Tensor predict(const std::vector<float>& state,
                         const cv::Mat& img_main,
                         const cv::Mat& img_secondary) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        torch::NoGradGuard no_grad;
        
        // Prepare state
        auto state_tensor = torch::from_blob(
            const_cast<float*>(state.data()),
            {static_cast<long>(state.size())},
            torch::kFloat32
        ).clone().to(config.device);
        
        state_tensor = normalize(state_tensor, "observation.state");
        state_tensor = state_tensor.unsqueeze(0); // Add batch dim
        
        // Prepare images
        auto img_main_tensor = preprocessImage(img_main).to(config.device);
        img_main_tensor = normalize(img_main_tensor, "observation.images.main");
        img_main_tensor = img_main_tensor.unsqueeze(0);
        
        auto img_secondary_tensor = preprocessImage(img_secondary).to(config.device);
        img_secondary_tensor = normalize(img_secondary_tensor, "observation.images.secondary_0");
        img_secondary_tensor = img_secondary_tensor.unsqueeze(0);
        
        // Create input dict
        std::vector<torch::jit::IValue> inputs;
        torch::Dict<std::string, torch::Tensor> input_dict;
        input_dict.insert("observation.state", state_tensor);
        input_dict.insert("observation.images.main", img_main_tensor);
        input_dict.insert("observation.images.secondary_0", img_secondary_tensor);
        inputs.push_back(input_dict);
        
        // Run inference
        auto output = model.forward(inputs).toTensor();
        
        // Unnormalize
        output = unnormalize(output, "action");
        
        // Track inference time
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        inference_times.push_back(elapsed);
        if (inference_times.size() > max_time_samples) {
            inference_times.pop_front();
        }
        
        return output.squeeze(0).cpu(); // Remove batch dim and move to CPU
    }
    
    double getAverageFPS() {
        if (inference_times.empty()) return 0.0;
        double avg_time = 0.0;
        for (double t : inference_times) {
            avg_time += t;
        }
        avg_time /= inference_times.size();
        return avg_time > 0 ? 1.0 / avg_time : 0.0;
    }
};

// Visualization class
class Visualizer {
private:
    std::vector<std::string> motor_names = {
        "motor_1", "motor_2", "motor_3", "motor_4",
        "motor_5", "motor_6", "motor_7"
    };
    
    cv::Mat canvas;
    const int img_width = 640;
    const int img_height = 480;
    const int plot_width = 400;
    const int plot_height = 150;
    
public:
    void visualize(const cv::Mat& img_main, const cv::Mat& img_secondary,
                  const std::vector<float>& state,
                  const torch::Tensor& predicted_actions,
                  const std::vector<float>& gt_action,
                  int frame_num, double fps) {
        
        // Create canvas
        int total_width = img_width * 2 + plot_width * 2;
        int total_height = img_height * 2 + plot_height * 2;
        canvas = cv::Mat(total_height, total_width, CV_8UC3, cv::Scalar(255, 255, 255));
        
        // Resize and place images
        cv::Mat img_main_resized, img_secondary_resized;
        cv::resize(img_main, img_main_resized, cv::Size(img_width, img_height));
        cv::resize(img_secondary, img_secondary_resized, cv::Size(img_width, img_height));
        
        img_main_resized.copyTo(canvas(cv::Rect(0, 0, img_width, img_height)));
        img_secondary_resized.copyTo(canvas(cv::Rect(0, img_height, img_width, img_height)));
        
        // Add titles
        std::string title = "ACT Inference - Frame " + std::to_string(frame_num) + 
                          " | FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(canvas, title, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
        
        cv::putText(canvas, "Main Camera", cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        cv::putText(canvas, "Secondary Camera", cv::Point(10, img_height + 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        
        // Plot action predictions
        auto pred_accessor = predicted_actions.accessor<float, 2>();
        int n_steps = predicted_actions.size(0);
        
        for (int motor_idx = 0; motor_idx < 7; motor_idx++) {
            int row = motor_idx / 2;
            int col = motor_idx % 2;
            int x_offset = img_width * 2 + col * plot_width;
            int y_offset = row * plot_height;
            
            plotMotorPrediction(motor_idx, state[motor_idx], 
                              pred_accessor, n_steps,
                              gt_action[motor_idx],
                              x_offset, y_offset);
        }
        
        // Display
        cv::imshow("ACT Policy Inference", canvas);
    }
    
private:
    void plotMotorPrediction(int motor_idx, float current_state,
                           const torch::TensorAccessor<float, 2>& predictions,
                           int n_steps, float gt_value,
                           int x_offset, int y_offset) {
        
        // Extract prediction values for this motor
        std::vector<float> values;
        float min_val = current_state, max_val = current_state;
        
        for (int i = 0; i < n_steps; i++) {
            float val = predictions[i][motor_idx];
            values.push_back(val);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        min_val = std::min({min_val, current_state, gt_value}) - 0.1f;
        max_val = std::max({max_val, current_state, gt_value}) + 0.1f;
        
        // Create plot area
        cv::Mat plot_area = canvas(cv::Rect(x_offset, y_offset, plot_width, plot_height));
        cv::rectangle(plot_area, cv::Point(0, 0), cv::Point(plot_width-1, plot_height-1),
                     cv::Scalar(200, 200, 200), 1);
        
        // Add title
        cv::putText(plot_area, motor_names[motor_idx], cv::Point(10, 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        
        // Plot area dimensions
        int margin = 30;
        int plot_w = plot_width - 2 * margin;
        int plot_h = plot_height - 2 * margin;
        
        // Draw predicted trajectory
        for (int i = 0; i < n_steps - 1; i++) {
            float x1 = margin + (i * plot_w) / (n_steps - 1);
            float y1 = plot_height - margin - ((values[i] - min_val) / (max_val - min_val)) * plot_h;
            float x2 = margin + ((i + 1) * plot_w) / (n_steps - 1);
            float y2 = plot_height - margin - ((values[i+1] - min_val) / (max_val - min_val)) * plot_h;
            
            cv::line(plot_area, cv::Point(x1, y1), cv::Point(x2, y2),
                    cv::Scalar(255, 0, 0), 2); // Blue for predicted
        }
        
        // Draw current state line
        float state_y = plot_height - margin - ((current_state - min_val) / (max_val - min_val)) * plot_h;
        cv::line(plot_area, cv::Point(margin, state_y), 
                cv::Point(plot_width - margin, state_y),
                cv::Scalar(0, 255, 0), 2); // Green for current
        
        // Draw ground truth line
        float gt_y = plot_height - margin - ((gt_value - min_val) / (max_val - min_val)) * plot_h;
        cv::line(plot_area, cv::Point(margin, gt_y),
                cv::Point(plot_width - margin, gt_y),
                cv::Scalar(0, 0, 255), 2); // Red for GT
        
        // Add legend
        cv::putText(plot_area, "Pred", cv::Point(10, plot_height - 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0), 1);
        cv::putText(plot_area, "Curr", cv::Point(10, plot_height - 15),
                   cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
        cv::putText(plot_area, "GT", cv::Point(10, plot_height - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1);
    }
};

// Simple CSV reader for parquet alternative (you'd need parquet C++ library for real parquet)
class DatasetReader {
private:
    std::vector<std::vector<float>> states;
    std::vector<std::vector<float>> actions;
    int current_idx = 0;
    
public:
    DatasetReader(const std::string& csv_path) {
        // Note: In production, use Apache Arrow C++ for parquet files
        // For now, assuming CSV export from parquet
        std::cout << "Note: For full functionality, export parquet to CSV or use Apache Arrow C++" << std::endl;
        std::cout << "Dataset path: " << csv_path << std::endl;
    }
    
    bool loadFromCSV(const std::string& csv_path) {
        std::ifstream file(csv_path);
        if (!file.is_open()) {
            std::cerr << "Could not open CSV file: " << csv_path << std::endl;
            return false;
        }
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string token;
            
            std::vector<float> state, action;
            
            // Parse state (7 values)
            for (int i = 0; i < 7; i++) {
                std::getline(iss, token, ',');
                state.push_back(std::stof(token));
            }
            
            // Parse action (7 values)
            for (int i = 0; i < 7; i++) {
                std::getline(iss, token, ',');
                action.push_back(std::stof(token));
            }
            
            states.push_back(state);
            actions.push_back(action);
        }
        
        std::cout << "Loaded " << states.size() << " samples from CSV" << std::endl;
        return true;
    }
    
    bool getFrame(int idx, std::vector<float>& state, std::vector<float>& action) {
        if (idx >= states.size()) return false;
        state = states[idx];
        action = actions[idx];
        return true;
    }
    
    int size() const { return states.size(); }
};

// Main continuous inference runner
int main(int argc, char* argv[]) {
    std::cout << "ACT Policy Continuous Inference (C++)" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // Paths (WSL format: /mnt/d instead of D:/)
    std::string checkpoint_dir = "/mnt/d/lerobot/lerobot/outputs/train/2025-11-20/19-50-21_act/checkpoints/300000";
    std::string dataset_dir = "/mnt/d/lerobot/lerobot/dataset/Marker_pickup_piper";
    
    // Parse command line args
    if (argc > 1) checkpoint_dir = argv[1];
    if (argc > 2) dataset_dir = argv[2];
    
    std::string video_main = dataset_dir + "/videos/observation.images.main/chunk-000/file-000.mp4";
    std::string video_secondary = dataset_dir + "/videos/observation.images.secondary_0/chunk-000/file-000.mp4";
    std::string data_csv = dataset_dir + "/data/chunk-000/data.csv"; // You need to export this
    
    std::cout << "Checkpoint directory: " << checkpoint_dir << std::endl;
    std::cout << "Dataset directory: " << dataset_dir << std::endl;
    
    // Initialize
    ACTInferenceConfig config(checkpoint_dir, dataset_dir, "cuda");
    ACTInference inference(config);
    Visualizer visualizer;
    
    // Open video captures
    cv::VideoCapture cap_main(video_main);
    cv::VideoCapture cap_secondary(video_secondary);
    
    if (!cap_main.isOpened() || !cap_secondary.isOpened()) {
        std::cerr << "Error opening video files!" << std::endl;
        return -1;
    }
    
    int total_frames = std::min(
        static_cast<int>(cap_main.get(cv::CAP_PROP_FRAME_COUNT)),
        static_cast<int>(cap_secondary.get(cv::CAP_PROP_FRAME_COUNT))
    );
    
    std::cout << "Total frames: " << total_frames << std::endl;
    std::cout << "Press 'q' to quit, 'p' to pause" << std::endl;
    
    // For demonstration, create dummy data if CSV doesn't exist
    // In production, load from CSV or use Apache Arrow for parquet
    bool use_dummy_data = true;
    
    // Main loop
    int frame_idx = 0;
    bool paused = false;
    
    while (frame_idx < total_frames) {
        if (!paused) {
            cv::Mat img_main, img_secondary;
            cap_main >> img_main;
            cap_secondary >> img_secondary;
            
            if (img_main.empty() || img_secondary.empty()) {
                std::cout << "End of video" << std::endl;
                break;
            }
            
            // Get state and action (dummy data for demo)
            std::vector<float> state(7);
            std::vector<float> gt_action(7);
            
            if (use_dummy_data) {
                // Generate dummy sinusoidal data for demonstration
                for (int i = 0; i < 7; i++) {
                    state[i] = std::sin(frame_idx * 0.01 + i) * 0.5;
                    gt_action[i] = std::sin((frame_idx + 1) * 0.01 + i) * 0.5;
                }
            }
            
            // Run inference
            auto predicted_actions = inference.predict(state, img_main, img_secondary);
            
            // Visualize
            double fps = inference.getAverageFPS();
            visualizer.visualize(img_main, img_secondary, state, 
                               predicted_actions, gt_action, frame_idx, fps);
            
            // Progress
            if (frame_idx % 100 == 0) {
                std::cout << "Frame " << frame_idx << "/" << total_frames 
                         << " | FPS: " << fps << std::endl;
            }
            
            frame_idx++;
        }
        
        // Handle keyboard input
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 'q' or ESC
            std::cout << "Quitting..." << std::endl;
            break;
        } else if (key == 'p') {
            paused = !paused;
            std::cout << (paused ? "Paused" : "Resumed") << std::endl;
        }
    }
    
    // Cleanup
    cap_main.release();
    cap_secondary.release();
    cv::destroyAllWindows();
    
    std::cout << "Inference complete!" << std::endl;
    return 0;
}

