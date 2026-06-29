#include "angular_momentum.h"
#include <cstring>

void launch();
void benchmark();

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
    return 0;
}