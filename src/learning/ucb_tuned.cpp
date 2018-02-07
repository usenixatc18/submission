#include "ucb_tuned.h"
#include <learning/average_func.h>
#include <learning/learning_method_factory_helper.h>

namespace {

using namespace schad;
using namespace learning;

template<class Average>
class UCBTunedLearningMethod : public LearningMethod {
    using Avg = AverageFunc<Average>;
    using AvgSq = AverageFunc<Average, true>;
public:
    UCBTunedLearningMethod(size_t num_arms, Average avg)
        : initial_arm_{0}, values_(num_arms, Avg{avg}), 
          squares_(num_arms, AvgSq{avg}) {
    }

    void report_rewards(vector<optional<Reward>> const& rewards) override {
        for (auto i = 0u; i < rewards.size(); ++i) {
            values_[i] += maybe_value(rewards[i]);
            squares_[i] += maybe_value(rewards[i]);
        }
    }


    auto choose() -> vector<size_t> override {
        if (initial_arm_ < values_.size()) {
            return {initial_arm_++};
        }

        auto const total_count = std::accumulate(begin(values_), end(values_), 0.0,
            [] (double x, auto const& y) {
                return x + y.count();
            }
        );

        auto const get_prio = [total_count,this] (auto i) {
            return values_[i].avg() + sqrt(log(total_count) / values_[i].count() * 
                std::min(0.25, 
                    squares_[i].avg() - values_[i].avg() * values_[i].avg() 
                    + sqrt(2.0 * log(total_count) / values_[i].count())
                )
            );
        };

        vector<size_t> result(values_.size());
        std::iota(begin(result), end(result), 0);
        std::sort(begin(result), end(result), [get_prio](auto a, auto b) {
            return get_prio(a) > get_prio(b);
        });
        return result;
    }

private:
    size_t initial_arm_;
    vector<Avg> values_;
    vector<AvgSq> squares_;
};

}

auto schad::learning::create_ucb_tuned(Average avg) 
    -> unique_ptr<LearningMethodFactory> {
    return std::visit([](auto&& avg) {
        using T = std::decay_t<decltype(avg)>;
        return create_learning_factory(
            [avg]([[maybe_unused]] auto rng, auto num_arms) {
                return make_unique<UCBTunedLearningMethod<T>>(num_arms, avg);
            },
            {{"type", "ucb_tuned"},{"parameters", {{"average", avg}}}}
        );
    }, avg);
}
