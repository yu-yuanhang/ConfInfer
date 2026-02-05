#include <iostream>

#include <trustinfer.h>
#include <ops.h>

int main(int argc, char *argv[]) {

    Conv2d conv1(1, 3, {3, 3});
    Conv2d conv2(3, 6, {3, 3});
    Conv2d conv3(6, 18, {3, 3});

    Value_t input;
    conv1(input);

    return 0;
}