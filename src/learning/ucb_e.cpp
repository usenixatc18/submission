#include "ucb_e.h"
#include <learning/learning_method_factory_helper.h>
#include <learning/average_func.h>

namespace {
using namespace schad;
using namespace learning;

class UCBExploration : public LearningMethod {
private:
    using Avg = AverageFunc<KeepAll>;
public:
    UCBExploration(double a, size_t num_arms) 
        : arm_value_(num_arms, Avg{KeepAll{}}), a_{a} {

    }

    auto choose() -> vector<size_t> override {
        auto get_prio = [this] (auto i) {
            return arm_value_[i].avg() + sqrt(a_ / arm_value_[i].count());
        };

        vector<size_t> result(arm_value_.size());
        std::iota(begin(result), end(result), 0);

        sort(begin(result), end(result), [get_prio](auto a, auto b) {
            return get_prio(a) > get_prio(b);
        });

        return result;
    }

    void report_rewards(vector<optional<Reward>> const& rewards) override {
        for (auto i = 0u; i < rewards.size(); ++i) {
            arm_value_[i] += maybe_value(rewards[i]);
        }
    }
private:
    vector<Avg> arm_value_;
    double const a_;
};

}

auto schad::learning::create_ucb_e(double a) 
        -> unique_ptr<LearningMethodFactory> {
    return create_learning_factory(
        [a] ([[maybe_unused]] auto rng, auto num_arms) {
            return make_unique<UCBExploration>(a, num_arms);
        },
        {{"type", "ucb_e"},{"parameters", {{"a", a}}}}
    );
}
