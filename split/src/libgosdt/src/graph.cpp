#include "graph.hpp"

void Graph::clear(void) {
    this->vertices.clear();
    this->edges.clear();
    this->translations.clear();
    this->children.clear();
    this->vertices.clear();
    this->bounds.clear();
    return;
}

unsigned int Graph::size(void) const { return this->vertices.size(); }
