#ifndef SIMULATION_H
#define SIMULATION_H

#include <simulator/experiment_parameters.h>
#include <simulator/statistics.h>
#include <common.h>


namespace schad {

auto run(ExperimentParameters const& parms) -> Statistics;

}

#endif // SIMULATION_H
