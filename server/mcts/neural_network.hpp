#pragma once

#include <vector>
#include <memory>
#include <torch/torch.h>
#include <torch/script.h>

class NeuralNetwork
{
public:
    NeuralNetwork(int gpu_id);

    // Single inference
    std::pair<std::vector<float>, float> predict(const std::vector<float> &state);

    // Batch inference
    std::pair<std::vector<float>, std::vector<float>> predict_batch(
        const std::vector<float> &states, int batch_size);

    // Load model from file
    void load_model(const std::string &path);

private:
    torch::Device device;
    torch::jit::script::Module model;

    torch::Tensor state_to_tensor(const std::vector<float> &state);
    torch::Tensor states_to_tensor(const std::vector<float> &states, int batch_size);
};
