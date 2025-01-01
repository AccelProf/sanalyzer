#include <iostream>


void yosemite_alloc_callback() {
    std::cout << "Memory allocated" << std::endl;
}

void yosemite_free_callback() {
    std::cout << "Memory freed" << std::endl;
}