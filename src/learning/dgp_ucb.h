#ifndef DGP_UCB_H
#define DGP_UCB_H

#include <learning/learning_method.h>
#include <learning/average.h>

namespace schad::learning {

auto create_dgp_ucb(double delta, double ksi, optional<Average> avg) 
    -> unique_ptr<LearningMethodFactory>;

}

#endif // DGP_UCB_H
