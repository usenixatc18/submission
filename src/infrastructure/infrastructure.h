#ifndef INFRASTRUCTURE_H
#define INFRASTRUCTURE_H

#include <policy/policy.h>
#include <learning/learning_method.h>
#include <reward/reward_function.h>
#include <infrastructure/infrastructure_parameters.h>
#include <infrastructure/stats_collector.h>
#include <common.h>

namespace schad {

struct Infrastructure {
    virtual auto active_policy() const 
        -> tuple<PolicyInstance *, RewardFunction *> = 0;
    virtual auto simulated_policies() const 
        -> vector<tuple<PolicyInstance *, RewardFunction *>> = 0;
    virtual void tick() = 0;

    Infrastructure() = default;
    Infrastructure(Infrastructure const&) = delete;
    Infrastructure& operator=(Infrastructure const&) = delete;
    
    virtual ~Infrastructure() = default;
};

auto create_infastructure(
    InfrastructureParameters const& params,
    RewardFunction const& reward,
    vector<shared_ptr<Policy>> const& policies,
    learning::LearningMethodFactory const& learning,
    World * world,
    StatsCollector * stat_collector
) -> unique_ptr<Infrastructure>;

}

#endif // INFRASTRUCTURE_H
