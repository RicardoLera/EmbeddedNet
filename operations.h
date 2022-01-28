#ifndef OP_H
#define OP_H

void convolution2D(int isize,  // width/height of input
        int osize,  // width/height of output
        int ksize,  // width/height of kernel
        int stride, // shift between input pixels, between consecutive outputs
        int pad,    // offset between (0,0) pixels between input and output
        int idepth, int odepth, // number of input and output channels
        float idata[isize][isize][idepth],
        float odata[osize][osize][odepth],
        float kdata[odepth][ksize][ksize][idepth]);

void batch_normalize(int size,
        int depth,
        float epsilon,
        float idata[size][size][depth],
        float odata[size][size][depth],
        float pdata[4][depth]);

void relu6(int size, int depth, float data[size][size][depth]);

void depthwise_convolution(int isize,
        int osize,
        int ksize,
        int stride,
        int pad,
        int depth,
        float idata[isize][isize][depth],
        float odata[osize][osize][depth],
        float kdata[ksize][ksize][depth]);

void pad(int size, int depth, float data[size][size][depth], float datap[size+1][size+1][depth]);

void add_array(int size, int depth, float idata[size][size][depth], float odata[size][size][depth]);

void avgpool(float data[7][7][1280], float pred[1280]);

void softmax(float *input, int input_len);

void fully_connect(int isize, int osize, float w[isize][osize], float *b, float *idata, float *odata);

void decode(float *pred);

void backprop_fc(float I[1280],
        float O[1000],
        int label,
        float weight[1280][1000],
        float bias[1000],
        float Ew[1280][1000],
        float Eb[1000]);


#endif
