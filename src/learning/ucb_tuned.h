#ifndef UCB_TUNED_H
#define UCB_TUNED_H

#include <learning/learning_method.h>
#include <learning/average.h>

namespace schad::learning {

auto create_ucb_tuned(Average avg) -> unique_ptr<LearningMethodFactory>;

}

#endif // UCB_TUNED_H
