#include "local_state.hpp"

LocalState::LocalState(void) {}

void LocalState::initialize(unsigned int _samples, unsigned int _features, unsigned int _targets) {
    this->samples = _samples;
    this->features = _features;
    this->targets = _targets;

    this->inbound_message.initialize(_samples, _features, _targets);
    this->outbound_message.initialize(_samples, _features, _targets);

    this->neighbourhood.resize(2 * (this->features));

    unsigned int buffer_count = 4;
    unsigned int row_size = this->features + this->targets;
    unsigned int column_size = this->samples;
    unsigned int max_tile_size = row_size * column_size;

    this->column_buffer = Bitmask(column_size);
}
