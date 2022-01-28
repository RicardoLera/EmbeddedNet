#include <stdlib.h>
#include <math.h>
#include "var.h"
#include "operations.h"
#include "data_manip.h"

//for testing
#include <stdio.h>

void conv2d(int isize,          // width/height of input
        int osize,              // width/height of output
        int ksize,              // width/height of kernel
        int stride,             // shift between input pixels, between consecutive outputs
        int pad,                // offset between (0,0) pixels between input and output
        int idepth, int odepth, // number of input and output channels
        int idx,                // weight import index
        float idata[isize][isize][idepth],
        float odata[osize][osize][odepth],
        float kdata[odepth][ksize][ksize][idepth])
{
    if (batch_count == 0) {
        // fill kernel using kdata [od][y][x][id] syntax
        import_weights(idepth,  // input depth (layers)
                ksize,          // sizes
                odepth,         // output depth (number of filters)
                idx,            // weight file index
                kdata);
    }

    convolution2D(isize, osize, ksize,  // input, output, kernel size
            stride,                     // stride
            pad,                        // pad input by one pixels de factor centering the kernel
            idepth, odepth,             // number of input and output channels
            idata, odata, kdata);
}

void batch_normalization(int size,  // width/height of input
        int depth,                  // number of channels
        int idx,                    // parameter import index
        float idata[size][size][depth],
        float odata[size][size][depth],
        float pdata[4][depth])
{
    if (batch_count == 0) {
        import_bn(depth,
                idx,
                pdata);
        if (!training) {
            // fill parameter data using pdata[type][x] syntax
            import_moving(depth,
                    idx,
                    pdata);
        }
    }

    batch_normalize(size,
            depth,
            0.001,  // epsilon
            idata, odata, pdata);
}

void S1depthwise (int size,
        int ksize,
        int pad,
        int depth,
        int idx,
        float idata[size][size][depth],
        float odata[size][size][depth],
        float kdata[ksize][ksize][depth])
{
    if (batch_count == 0) {
        // fill kernel using kdata [y][x][d] syntax
        import_dweights(depth,
                ksize,
                idx,
                kdata);
    }

    depthwise_convolution(size, size, ksize,    // input, output, kernel size
            1,                                  // stride
            pad,                                // pad input by one pixels de factor centering the kernel
            depth,                              // number of channels
            idata, odata, kdata);
}

void S2depthwise (int isize,
        int osize,
        int ksize,
        int pad,
        int depth,
        int idx,
        float idata[isize][isize][depth],
        float odata[osize][osize][depth],
        float kdata[ksize][ksize][depth])
{
    if (batch_count == 0) {
        // fill kernel using kdata [y][x][d] syntax
        import_dweights(depth,
                ksize,
                idx,
                kdata);
    }

    depthwise_convolution(isize, osize, ksize,  // input, output, kernel size
            2,                                  // stride
            pad,                                // pad input by one pixels de factor centering the kernel
            depth,                              // number of channels
            idata, odata, kdata);
}

void fully_connected(float *data, float *pred, float weight[1280][1000], float bias[1000]) {

    if (batch_count == 0) {
        import_fc(1280,
                1000,
                weight,
                bias);
    }

    fully_connect(1280,
            1000,
            weight,
            bias,
            data,
            pred);
}
