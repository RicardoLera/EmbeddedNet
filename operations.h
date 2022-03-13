#ifndef OP_H
#define OP_H

void convolution2D(int isize,  // width/height of input
        int osize,  // width/height of output
        int ksize,  // width/height of kernel
        int stride, // shift between input pixels, between consecutive outputs
        int pad,    // offset between (0,0) pixels between input and output
        int idepth, int odepth, // number of input and output channels
        float idata[restrict isize][isize][idepth],
        float odata[restrict osize][osize][odepth],
        float kdata[restrict odepth][ksize][ksize][idepth]);

void batch_normalize(int size,
        int depth,
        float epsilon,
        float idata[restrict size][size][depth],
        float odata[restrict size][size][depth],
        float pdata[restrict 4][depth]);

void relu6(int size, int depth, float data[size][size][depth]);

void depthwise_convolution(int isize,
        int osize,
        int ksize,
        int stride,
        int pad,
        int depth,
        float idata[restrict isize][isize][depth],
        float odata[restrict osize][osize][depth],
        float kdata[restrict ksize][ksize][depth]);

void pad(int size, int depth, float data[size][size][depth], float datap[size+1][size+1][depth]);

void add_array(int size, int depth, float idata[size][size][depth], float odata[size][size][depth]);

void avgpool(float data[restrict 7][7][1280], float pred[1280]);

void softmax(float *input, int input_len);

void fully_connect(int isize, int osize, float w[restrict isize][osize][2], float b[restrict osize][2], float *idata, float *odata);

void decode(float *pred);

void backprop_fc(int c,
        float I[1280],
        float O[c],
        int label,
        float fc_w[restrict 1280][c][2],
        float fc_b[restrict c][2]);

void backprop_avrgpool(float I[restrict 7][7][1280], float O[1280]);

void backprop_relu6(int s, int d, float I[restrict s][s][d]);

void backprop_bn(int s, int d,
        float I[restrict s][s][d], float O[restrict s][s][d],
        float par[restrict 4][d], float Ep[restrict 4][d],
        int idx);

void backprop_conv2d(int isize, int osize, int ksize, int idepth, int odepth,
        float I[restrict isize][isize][idepth], float O[restrict osize][osize][odepth],
        float par[restrict odepth][ksize][ksize][idepth], float Ew[restrict odepth][ksize][ksize][idepth],
        int stride, int pad, int idx);

void backprop_dw(int isize, int osize, int ksize, int depth,
        float I[restrict isize][isize][depth], float O[restrict osize][osize][depth],
        float par[restrict ksize][ksize][depth], float Ew[restrict ksize][ksize][depth],
        int stride, int pad, int idx);

float nat_log (float n);

void loss_plot (int epoch_count);

#endif
