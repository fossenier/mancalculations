#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mcts.hpp"
#include "kalah_game.hpp"

namespace py = pybind11;

// Python wrapper for KalahGame
class PyKalahGame : public KalahGame
{
public:
    using KalahGame::KalahGame;

    py::array_t<float> get_canonical_state_py() const
    {
        auto state = get_canonical_state();
        return py::array_t<float>(state.size(), state.data());
    }

    py::array_t<bool> get_valid_moves_py() const
    {
        auto moves = get_valid_moves();
        return py::array_t<bool>(moves.size(), moves.data());
    }
};

// Python wrapper for MCTS
class PyMCTS : public MCTS
{
public:
    PyMCTS(py::dict config, const std::string &model_path,
           int num_threads = 64, int num_gpus = 4)
        : MCTS(parse_config(config), num_threads, num_gpus), model_path_(model_path)
    {

        // Load the model on all GPUs
        for (auto &gpu_proc : gpu_processors_)
        {
            // You'll need to implement model loading in GPUBatchProcessor
        }
    }

    py::array_t<float> search_py(const PyKalahGame &game, const std::string &root_state = "")
    {
        auto visits = search(game, root_state);
        return py::array_t<float>(visits.size(), visits.data());
    }

    py::array_t<float> get_action_probabilities_py(const PyKalahGame &game, float temperature = 1.0f)
    {
        auto probs = get_action_probabilities(game, temperature);
        return py::array_t<float>(probs.size(), probs.data());
    }

private:
    std::string model_path_;

    static MCTSConfig parse_config(py::dict config)
    {
        MCTSConfig mcts_config;

        if (config.contains("mcts"))
        {
            auto mcts = config["mcts"];
            if (mcts.contains("num_simulations"))
                mcts_config.num_simulations = mcts["num_simulations"].cast<int>();
            if (mcts.contains("c_puct"))
                mcts_config.c_puct = mcts["c_puct"].cast<float>();
            if (mcts.contains("dirichlet_alpha"))
                mcts_config.dirichlet_alpha = mcts["dirichlet_alpha"].cast<float>();
            if (mcts.contains("dirichlet_epsilon"))
                mcts_config.dirichlet_epsilon = mcts["dirichlet_epsilon"].cast<float>();
            if (mcts.contains("temperature_threshold"))
                mcts_config.temperature_threshold = mcts["temperature_threshold"].cast<int>();
            if (mcts.contains("initial_temperature"))
                mcts_config.initial_temperature = mcts["initial_temperature"].cast<float>();
            if (mcts.contains("final_temperature"))
                mcts_config.final_temperature = mcts["final_temperature"].cast<float>();
        }

        return mcts_config;
    }
};

PYBIND11_MODULE(mcts_cpp, m)
{
    m.doc() = "C++ MCTS implementation for Kalah";

    // Bind KalahGame
    py::class_<PyKalahGame>(m, "KalahGame")
        .def(py::init<>())
        .def("make_move", &PyKalahGame::make_move)
        .def("is_game_over", &PyKalahGame::is_game_over)
        .def("get_reward", &PyKalahGame::get_reward)
        .def("get_valid_moves", &PyKalahGame::get_valid_moves_py)
        .def("get_canonical_state", &PyKalahGame::get_canonical_state_py)
        .def("get_current_player", &PyKalahGame::get_current_player)
        .def("clone", &PyKalahGame::clone)
        .def_property_readonly("game_over", &PyKalahGame::is_game_over)
        .def_property_readonly("current_player", &PyKalahGame::get_current_player);

    // Bind MCTS
    py::class_<PyMCTS>(m, "MCTS")
        .def(py::init<py::dict, const std::string &, int, int>(),
             py::arg("config"),
             py::arg("model_path"),
             py::arg("num_threads") = 64,
             py::arg("num_gpus") = 4)
        .def("search", &PyMCTS::search_py,
             py::arg("game"),
             py::arg("root_state") = "")
        .def("get_action_probabilities", &PyMCTS::get_action_probabilities_py,
             py::arg("game"),
             py::arg("temperature") = 1.0f)
        .def("clear_tree", &PyMCTS::clear_tree);
}
