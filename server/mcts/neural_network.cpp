#include "neural_network.hpp"
#include <torch/cuda.h>

NeuralNetwork::NeuralNetwork(int gpu_id)
    : device(torch::cuda::is_available() && gpu_id >= 0
                 ? torch::Device(torch::kCUDA, gpu_id)
                 : torch::Device(torch::kCPU))
{

    // Set CUDA device
    if (device.is_cuda())
    {
        torch::cuda::set_device(device.index());
    }
}

void NeuralNetwork::load_model(const std::string &path)
{
    try
    {
        model = torch::jit::load(path, device);
        model.eval();

        // Optimize for inference
        if (device.is_cuda())
        {
            model = torch::jit::optimize_for_inference(model);
        }
    }
    catch (const c10::Error &e)
    {
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
}

torch::Tensor NeuralNetwork::state_to_tensor(const std::vector<float> &state)
{
    return torch::from_blob(
               const_cast<float *>(state.data()),
               {1, static_cast<long>(state.size())},
               torch::kFloat32)
        .to(device);
}

torch::Tensor NeuralNetwork::states_to_tensor(const std::vector<float> &states, int batch_size)
{
    int state_size = states.size() / batch_size;
    return torch::from_blob(
               const_cast<float *>(states.data()),
               {batch_size, state_size},
               torch::kFloat32)
        .to(device);
}

std::pair<std::vector<float>, float> NeuralNetwork::predict(const std::vector<float> &state)
{
    torch::NoGradGuard no_grad;

    auto input = state_to_tensor(state);
    auto outputs = model.forward({input}).toTuple();

    auto policy = outputs->elements()[0].toTensor().cpu();
    auto value = outputs->elements()[1].toTensor().cpu();

    // Convert to vectors
    std::vector<float> policy_vec(policy.data_ptr<float>(),
                                  policy.data_ptr<float>() + policy.numel());
    float value_scalar = value.item<float>();

    return {policy_vec, value_scalar};
}

std::pair<std::vector<float>, std::vector<float>> NeuralNetwork::predict_batch(
    const std::vector<float> &states, int batch_size)
{

    torch::NoGradGuard no_grad;

    auto input = states_to_tensor(states, batch_size);
    auto outputs = model.forward({input}).toTuple();

    auto policies = outputs->elements()[0].toTensor().cpu();
    auto values = outputs->elements()[1].toTensor().cpu();

    // Convert to vectors
    std::vector<float> policy_vec(policies.data_ptr<float>(),
                                  policies.data_ptr<float>() + policies.numel());
    std::vector<float> value_vec(values.data_ptr<float>(),
                                 values.data_ptr<float>() + values.numel());

    return {policy_vec, value_vec};
}
