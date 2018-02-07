#ifndef LEARNING_CONFIG_H
#define LEARNING_CONFIG_H

#include <learning/learning_method.h>
#include <configs/loader.h>

namespace schad {

struct unknown_learning_method_exception : std::invalid_argument {
    unknown_learning_method_exception(string const& name)
        : invalid_argument("unknown learning method: " + name) {
    }
};

auto load_learning_method(json const& cfg, Loader const& loader) 
    -> shared_ptr<learning::LearningMethodFactory>;

}

#endif // LEARNING_CONFIG_H
