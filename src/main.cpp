#include "segprefix.h"
#include <vector>
#include <iostream>


int main() {
    std::vector<int> values{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int> groups{0, 2, 5, 9};
    segmentedPrefixSum(values, groups);
    for (int v : values)
        std::cout << v << " ";
    std::cout << std::endl;
    return 0;
}

