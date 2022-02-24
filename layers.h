#ifndef LAYERS_H
#define LAYERS_H

void conv2d(int isize,          // width/height of input
        int osize,              // width/height of output
        int ksize,              // width/height of kernel
        int stride,             // shift between input pixels, between consecutive outputs
        int pad,                // offset between (0,0) pixels between input and output
        int idepth, int odepth, // number of input and output channels
        int idx,                // weight import index
        float idata[isize][isize][idepth],
        float odata[osize][osize][odepth],
        float kdata[odepth][ksize][ksize][idepth]);

void batch_normalization(int size,  // width/height of input
        int depth,                  // number of channels
        int idx,                    // parameter import index
        float idata[size][size][depth],
        float odata[size][size][depth],
        float pdata[4][depth]);

void S1depthwise (int size,
        int ksize,
        int pad,
        int depth,
        int idx,
        float idata[size][size][depth],
        float odata[size][size][depth],
        float kdata[ksize][ksize][depth]);

void S2depthwise (int isize,
        int osize,
        int ksize,
        int pad,
        int depth,
        int idx,
        float idata[isize][isize][depth],
        float odata[osize][osize][depth],
        float kdata[ksize][ksize][depth]);

void fully_connected(int c, float *data, float *pred, float weight[1280][c][2], float bias[c][2]);

#endif
