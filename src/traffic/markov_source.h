#ifndef MARKOV_SOURCE_H
#define MARKOV_SOURCE_H

#include <traffic/source.h>
#include <traffic/markov_source_parameters.h>

namespace schad {

auto create_markov_source(MarkovSourceParameters params) -> unique_ptr<SourceFactory>;  

}

#endif
