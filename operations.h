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

void fully_connect(int isize, int osize, float w[isize][osize][2], float b[osize][2], float *idata, float *odata);

void decode(float *pred);

void backprop_fc(int c,
        float I[1280],
        float O[c],
        int label,
        float fc_w[1280][c][2],
        float fc_b[c][2]);

void backprop_avrgpool(float I[7][7][1280], float O[1280]);

void backprop_relu6(int s, int d, float I[s][s][d]);

void backprop_bn(int s, int d,
        float I[s][s][d], float O[s][s][d],
        float par[4][d], float Ep[4][d],
        int idx);

void backprop_conv2d(int isize, int osize, int ksize, int idepth, int odepth,
        float I[isize][isize][idepth], float O[osize][osize][odepth],
        float par[odepth][ksize][ksize][idepth], float Ew[odepth][ksize][ksize][idepth],
        int stride, int pad, int idx);

void backprop_dw(int isize, int osize, int ksize, int depth,
        float I[isize][isize][depth], float O[osize][osize][depth],
        float par[ksize][ksize][depth], float Ew[ksize][ksize][depth],
        int stride, int pad, int idx);

#endif
