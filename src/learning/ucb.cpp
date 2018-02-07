#include "ucb.h"
#include <learning/average_func.h>
#include <learning/learning_method_factory_helper.h>

namespace {
using namespace schad;
using namespace learning;

template<class Average, bool restricted_exploration>
class UCBLearningMethod : public LearningMethod {
private:
    using Avg = AverageFunc<Average>;

public:
    UCBLearningMethod(double ksi, Average avg, size_t num_arms)
        : ksi_{ksi}, init_arm_{0}, arm_value_(num_arms, Avg{avg}),
         last_choice_{0}, my_value_(num_arms, Avg{avg}) {
    }

    auto choose() -> vector<size_t> override {
        if (init_arm_ < arm_value_.size()) {
            if constexpr (restricted_exploration) {
                last_choice_ = init_arm_;
            }
            return {init_arm_++};
        }

        auto const& exp_count = [this]() -> auto const& {
            if constexpr (restricted_exploration) {
                return  my_value_;
            } else {
                return arm_value_;
            }
        }();

        auto const total_exp_count = std::accumulate(
            begin(exp_count), end(exp_count), 0.0,
            [] (double x, auto const & y) {
                return x + y.count();
            }
        );

        auto get_prio = [this,total_exp_count,&exp_count] (auto i) {
                return arm_value_[i].avg() 
                    + sqrt(ksi_ * log(total_exp_count) / exp_count[i].count());
        };

        vector<size_t> result(arm_value_.size());
        std::iota(begin(result), end(result), 0);

        sort(begin(result), end(result), [get_prio](auto a, auto b) {
            return get_prio(a) > get_prio(b);
        });

        if constexpr(restricted_exploration) {
            last_choice_ = result.front();
        }

        return result;
    }

    void report_rewards(vector<optional<Reward>> const& rewards) override {
        if constexpr(restricted_exploration) {
            my_value_[last_choice_] += maybe_value(rewards[last_choice_]);
        } else {
            for (auto i = 0u; i < rewards.size(); ++i) {
                arm_value_[i] += maybe_value(rewards[i]);
            }
        }
    }

private:
    double const ksi_;
    size_t init_arm_;
    vector<Avg> arm_value_;

    size_t last_choice_;
    vector<Avg> my_value_;
};

}

auto schad::learning::create_ucb(UCBParameters const& params) 
        -> unique_ptr<LearningMethodFactory> {
    return create_learning_factory([params=params]
        (auto rng, auto num_arms) {
            return std::visit([rng,num_arms,&params] (auto&& avg) -> unique_ptr<LearningMethod> {
                using T = std::decay_t<decltype(avg)>;
                if (params.restricted_exploration()) {
                    return make_unique<UCBLearningMethod<T,true>>(params.ksi(), avg, num_arms);
                } else {
                    return make_unique<UCBLearningMethod<T,false>>(params.ksi(), avg, num_arms);
                }
            }, params.average());
        },
        {{"type", "ucb"}, {"parameters", params}}
    );
}
