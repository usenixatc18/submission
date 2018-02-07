#include <learning/constant.h>
#include <learning/ucb.h>
#include <learning/epsilon_greedy.h>
#include <learning/combined.h>
#include <learning/restarting.h>
#include <learning/ucb_e.h>
#include <learning/dgp_ucb.h>
#include <learning/softmax.h>
#include <learning/normalized.h>
#include <learning/ucb_tuned.h>
#include <learning/scaled.h>
#include "learning_config.h"

namespace schad::learning {
    void from_json(json const& average, learning::Average& avg) {
        if (average.at("type") == "keep_all") {
            avg = learning::KeepAll{};
        } else if (average.at("type") == "exponential") {
            avg = learning::Exponential{average.at("gamma").get<double>()};
        } else if (average.at("type") == "sliding_window") {
            avg = learning::SlidingWindow{average.at("num_steps").get<size_t>()};
        } else {
            throw std::invalid_argument("Unknown UCB averager!");
        }
    }
}

auto schad::load_learning_method(json const& cfg, Loader const& loader) 
    -> shared_ptr<learning::LearningMethodFactory> {
    auto const type = cfg.at("type").get<string>();
    auto const& params = cfg.at("parameters");
    if (type == "ucb") {
        return learning::create_ucb(learning::UCBParameters{}
            .set_ksi(params.at("ksi"))
            .set_average(params.at("average"))
            .set_restricted_exploration(params.at("restricted_exploration"))
        );
    } else if (type == "epsilon_greedy") {
        return learning::create_epsilon_greedy(
            learning::EpsilonGreedyParameters{}
                .set_epsilon(params.at("epsilon"))
                .set_average(params.at("average"))
                .set_delta(params.at("delta"))
                .set_sigma(params.at("sigma"))
        );
    } else if (type == "constant") {
        return learning::create_constant(params.at("arm_idx"));
    } else if (type == "combined") {
        return learning::create_combined(
            loader.load(load_learning_method, params.at("exploiter")),
            loader.load(load_learning_method, params.at("explorer"))
        );
    } else if (type == "restarting") {
        return learning::create_restarting(
            loader.load(load_learning_method, params.at("base")),
            params.at("num_steps")
        );
    } else if (type == "ucb_e") {
        return learning::create_ucb_e(params.at("a"));
    } else if (type == "dgp_ucb") {
        optional<learning::Average> average{};
        if (!params.at("average").is_null()) {
            average = params.at("average").get<learning::Average>();
        }
        return learning::create_dgp_ucb(
            params.at("delta"), params.at("ksi"), average
        );
    } else if (type == "softmax") {
        return learning::create_softmax(params.at("alpha"));
    } else if (type == "normalized") {
        return learning::create_normalized(
                loader.load(load_learning_method, params.at("base")));
    } else if (type == "ucb_tuned") {
        return learning::create_ucb_tuned(params.at("average"));
    } else if (type == "scaled") {
        return learning::create_scaled(
            loader.load(load_learning_method, params.at("base")),
            params.at("factor")
        );
    } else {
        throw unknown_learning_method_exception(type);
    }
}
