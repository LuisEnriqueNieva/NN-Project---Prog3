#pragma once

#include "PatternClassifier.h"

namespace utec::tf::apps {

class ControllerDemo {
private:
    PatternClassifier classifier_;
public:
    PatternClassifier& classifier() { return classifier_; }
};

} // namespace utec::tf::apps
