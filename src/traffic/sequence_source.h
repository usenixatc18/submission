#ifndef SEQUENCE_SOURCE_H
#define SEQUENCE_SOURCE_H

#include <traffic/source.h>
#include <traffic/sequence_source_parameters.h>

namespace schad {

auto create_sequence_source(SequenceSourceParameters params) -> unique_ptr<SourceFactory>;  

}

#endif
