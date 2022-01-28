#include <math.h>
#include <stdlib.h>
#include "var.h"
#include "actions.h"
#include "data_manip.h"

// for testing
#include <stdio.h>

#define LAMBDA 0.00004
#define EPSILON 0.0000001
#define LR 0.045
#define LR_DECAY 0.98
#define E_MOMENTUM 0.9
#define E_DECAY 0.9

void convolution2D(int isize,  // width/height of input
        int osize,  // width/height of output
        int ksize,  // width/height of kernel
        int stride, // shift between input pixels, between consecutive outputs
        int pad,    // offset between (0,0) pixels between input and output
        int idepth, int odepth, // number of input and output channels
        float idata[isize][isize][idepth],
        float odata[osize][osize][odepth],
        float kdata[odepth][ksize][ksize][idepth])
{
    // iterate over the output
    for (int oy = 0; oy < osize; ++oy) {
    for (int ox = 0; ox < osize; ++ox) {
    for (int od = 0; od < odepth; ++od) {
        for (int ky = 0; ky < ksize; ++ky) {
        for (int kx = 0; kx < ksize; ++kx) {
            // map position in output and kernel to the input
            int iy = stride * oy + ky - pad;
            int ix = stride * ox + kx - pad;
            // use only valid inputs
            if (iy >= 0 && iy < isize && ix >= 0 && ix < isize)
                for (int id = 0; id < idepth; ++id)
                    odata[oy][ox][od] += kdata[od][ky][kx][id] * idata[iy][ix][id];
        }}
    }}}

}

void batch_normalize(int size,
        int depth,
        float epsilon,
        float idata[size][size][depth],
        float odata[size][size][depth],
        float pdata[4][depth])
{
    for (int d = 0; d < depth; ++d)
    {
        // [0][:] is gamma, [1][:] is beta, [2][:] is moving mean, [3][:] is moving variance
        // Here we use moving mean and variance instead of calculating them, as opposed to training.
        for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            idata[y][x][d] = (idata[y][x][d] - pdata[2][d]) / sqrt(pdata[3][d] + epsilon); // normalize
            odata[y][x][d] = (idata[y][x][d] * pdata[0][d]) + pdata[1][d]; // scale and shift
        }}
    }
}

void relu6(int size, int depth, float data[size][size][depth])
{
    for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
    for (int d = 0; d < depth; ++d) {
        if (data[y][x][d] > 6) {data[y][x][d] = 6;}
        if (data[y][x][d] < 0) {data[y][x][d] = 0;}
    }}}
}

void depthwise_convolution(int isize,
        int osize,
        int ksize,
        int stride,
        int pad,
        int depth,
        float idata[isize][isize][depth],
        float odata[osize][osize][depth],
        float kdata[ksize][ksize][depth])
{
    // iterate over the output
    for (int oy = 0; oy < osize; ++oy) {
    for (int ox = 0; ox < osize; ++ox) {
    for (int od = 0; od < depth; ++od) {
        for (int ky = 0; ky < ksize; ++ky) {
        for (int kx = 0; kx < ksize; ++kx) {
                // map position in output and kernel to the input
                int iy = stride * oy + ky - pad;
                int ix = stride * ox + kx - pad;
                // use only valid inputs
                if (iy >= 0 && iy < isize && ix >= 0 && ix < isize)
                    odata[oy][ox][od] += kdata[ky][kx][od] * idata[iy][ix][od];
        }}
    }}}

}

void pad(int size, int depth, float data[size][size][depth], float datap[size+1][size+1][depth])
{
    for (int y = 0; y < size+1; ++y) {
    for (int x = 0; x < size+1; ++x) {
    for (int d = 0; d < depth; ++d) {
        if (y == 0 || y == size || x == 0 || x == size) {
            datap[y][x][d] = 0;
        }
        else
        {
            datap[y][x][d] = data[y][x][d];
        }
    }}}

}

void add_array(int size, int depth, float idata[size][size][depth], float odata[size][size][depth])
{
    for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
    for (int d = 0; d < depth; ++d) {
        odata[y][x][d] = odata[y][x][d] + idata[y][x][d];
    }}}

}

void avgpool(float data[7][7][1280], float pred[1280])
{
    for (int d = 0; d < 1280; ++d) {
        float avg = 0;
        for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 7; ++x) {
                avg = avg + data[y][x][d];
        }}
        pred[d] = avg / 49;
    }
}

void softmax(float *input, int input_len) {

    float m = -INFINITY;
    for (int i = 0; i < input_len; i++) {
        if (input[i] > m) {
            m = input[i];
        }
    }

    float sum = 0.0;
    for (int i = 0; i < input_len; i++) {
        sum += expf(input[i] - m);
    }

    float offset = m + logf(sum);
    for (int i = 0; i < input_len; i++) {
        input[i] = expf(input[i] - offset);
    }
}

void fully_connect(int isize, int osize, float w[isize][osize], float *b, float *idata, float *odata) {

    for (int y = 0; y < osize; ++y) { // 1000
        odata[y] = 0;
        for (int x = 0; x < isize; ++x) { // 1280
            odata[y] += idata[x]*w[x][y];
        }
        odata[y] += b[y];
    }
}

struct Correlation {
    float prediction;
    int idx;
};

int compare(const void *p1, const void *p2)
{
    const struct Correlation *elem1 = p1;
    const struct Correlation *elem2 = p2;

    if (elem1->prediction < elem2->prediction)
        return 1;
    else if (elem1->prediction > elem2->prediction)
        return -1;
    else
        return 0;
}

void decode(float *pred) {

    struct Correlation cor[1000];

    for(int n = 0; n < 1000; n++) {
        cor[n].prediction = pred[n];
        cor[n].idx = n;
    }

    qsort(cor, 1000, sizeof(struct Correlation), compare);

    FILE *fp;
    fp = fopen("fc_pred.csv", "w");
    if (fp == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "%.7e\n%d\n", cor[0].prediction, cor[0].idx);
    fclose(fp);
    printf("Saved fc_pred.csv\n");

    printf("\n");
    for (int n = 0; n < 5; n++)
        printf("Correlation %d: %e / Index = %d\n", n+1, cor[n].prediction, cor[n].idx);
    printf("\n");

    FILE *fptr;
    fptr = fopen("labels.txt", "r");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    char s[122];
    int j;

    for (int n = 0; n < 5; n++)
    {
        char c = fgetc(fptr);
        for (int i = 0; i < cor[n].idx; ++i)
        {
            while(c != '\n') {c = fgetc(fptr);} // skip line
            c = fgetc(fptr);                    // go to new line
        }
        while (c != '\''){c = fgetc(fptr);}      // go to start of phrase
        c = fgetc(fptr);
        for (j = 0; c != '\''; ++j){
            s[j] = c;
            c = fgetc(fptr);
        }
        s[j] = '\0';
        printf("'%s', %.8e\n", s, cor[n].prediction);
        rewind(fptr);
    }
    fclose(fptr);
}

void batch_sum(int d1, int d2, float data[d1][d1][d2], float temp_img[d1][d1][d2][96]) {
    for (int b = 0; b < 96; ++b) {
        inference(batch[b].image);
        for (int k = 0; k < d2; ++k) {
        for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d1; ++j) {
            temp_sum[k] += data[i][j][k];
        }}}
    }
}

void mean(int d, float data[4][d]) {
    for (int k = 0; k < d; ++k) {
        data[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
}

void batch_variance(int d1, int d2, float data[d1][d1][d2], float kdata[4][d2]) {
    for (int b = 0; b < 96; ++b) {
        inference(batch[b].image);
        for (int k = 0; k < d2; ++k) {
        for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d1; ++j) {
            temp_variance[k] += (data[i][j][k] - kdata[2][k]) * (data[i][j][k] - kdata[2][k]);
        }}}
    }
}

void variance(int d, float data[4][d]) {
    for (int k = 0; k < d; ++k) {
        data[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
}

void batch_calculate() {

    // Initial
    stop = 1;
    batch_sum(112, 32, initial_conv2d);
    mean(32, par[0].initial_par_BN);
    batch_variance(112, 32, initial_conv2d, par[0].initial_par_BN);
    variance(32, par[0].initial_par_BN);

    stop = 2;
    batch_sum(112, 32, initial_conv2d);
    mean(32, par[0].initial_par_BN);
    batch_variance(112, 32, initial_conv2d, par[0].initial_par_BN);
    variance(32, par[0].initial_par_BN);


    // Initial
    stop = 1;
    all_inferences();
    for (int k = 0; k < 32; ++k) {
        par[0].initial_par_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 2;
    all_inferences();
    for (int k = 0; k < 32; ++k) {
        par[0].initial_par_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Expanded
    stop = 3;
    all_inferences();
    for (int k = 0; k < 32; ++k) {
        par[0].expanded_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 4;
    all_inferences();
    for (int k = 0; k < 32; ++k) {
        par[0].expanded_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 5;
    all_inferences();
    for (int k = 0; k < 16; ++k) {
        par[0].expanded_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 6;
    all_inferences();
    for (int k = 0; k < 16; ++k) {
        par[0].expanded_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 3
    stop = 7;
    all_inferences();
    for (int k = 0; k < 96; ++k) {
        par[0].block3_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 8;
    all_inferences();
    for (int k = 0; k < 96; ++k) {
        par[0].block3_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 9;
    all_inferences();
    for (int k = 0; k < 96; ++k) {
        par[0].block3_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 10;
    all_inferences();
    for (int k = 0; k < 96; ++k) {
        par[0].block3_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 11;
    all_inferences();
    for (int k = 0; k < 24; ++k) {
        par[0].block3_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 12;
    all_inferences();
    for (int k = 0; k < 24; ++k) {
        par[0].block3_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 4
    stop = 13;
    all_inferences();
    for (int k = 0; k < 144; ++k) {
        par[0].block4_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 14;
    all_inferences();
    for (int k = 0; k < 144; ++k) {
        par[0].block4_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 15;
    all_inferences();
    for (int k = 0; k < 144; ++k) {
        par[0].block4_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 16;
    all_inferences();
    for (int k = 0; k < 144; ++k) {
        par[0].block4_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 17;
    all_inferences();
    for (int k = 0; k < 24; ++k) {
        par[0].block4_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 18;
    all_inferences();
    for (int k = 0; k < 24; ++k) {
        par[0].block4_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 5
    stop = 19;
    all_inferences();
    for (int k = 0; k < 144; ++k) {
        par[0].block5_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 20;
    all_inferences();
    for (int k = 0; k < 144; ++k) {
        par[0].block5_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 21;
    all_inferences();
    for (int k = 0; k < 144; ++k) {
        par[0].block5_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 22;
    all_inferences();
    for (int k = 0; k < 144; ++k) {
        par[0].block5_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 23;
    all_inferences();
    for (int k = 0; k < 32; ++k) {
        par[0].block5_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 24;
    all_inferences();
    for (int k = 0; k < 32; ++k) {
        par[0].block5_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 6
    stop = 25;
    all_inferences();
    for (int k = 0; k < 192; ++k) {
        par[0].block6_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 26;
    all_inferences();
    for (int k = 0; k < 192; ++k) {
        par[0].block6_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 27;
    all_inferences();
    for (int k = 0; k < 192; ++k) {
        par[0].block6_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 28;
    all_inferences();
    for (int k = 0; k < 192; ++k) {
        par[0].block6_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 29;
    all_inferences();
    for (int k = 0; k < 32; ++k) {
        par[0].block6_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 30;
    all_inferences();
    for (int k = 0; k < 32; ++k) {
        par[0].block6_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 7
    stop = 31;
    all_inferences();
    for (int k = 0; k < 192; ++k) {
        par[0].block7_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 32;
    all_inferences();
    for (int k = 0; k < 192; ++k) {
        par[0].block7_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 33;
    all_inferences();
    for (int k = 0; k < 192; ++k) {
        par[0].block7_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 34;
    all_inferences();
    for (int k = 0; k < 192; ++k) {
        par[0].block7_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 35;
    all_inferences();
    for (int k = 0; k < 32; ++k) {
        par[0].block7_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 36;
    all_inferences();
    for (int k = 0; k < 32; ++k) {
        par[0].block7_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 8
    stop = 37;
    all_inferences();
    for (int k = 0; k < 192; ++k) {
        par[0].block8_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 38;
    all_inferences();
    for (int k = 0; k < 192; ++k) {
        par[0].block8_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 39;
    all_inferences();
    for (int k = 0; k < 192; ++k) {
        par[0].block8_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 40;
    all_inferences();
    for (int k = 0; k < 192; ++k) {
        par[0].block8_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 41;
    all_inferences();
    for (int k = 0; k < 64; ++k) {
        par[0].block8_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 42;
    all_inferences();
    for (int k = 0; k < 64; ++k) {
        par[0].block8_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 9
    stop = 43;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block8_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 44;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block8_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 45;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block8_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 46;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block8_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 47;
    all_inferences();
    for (int k = 0; k < 64; ++k) {
        par[0].block8_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 48;
    all_inferences();
    for (int k = 0; k < 64; ++k) {
        par[0].block8_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 10
    stop = 49;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block10_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 50;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block10_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 51;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block10_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 52;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block10_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 53;
    all_inferences();
    for (int k = 0; k < 64; ++k) {
        par[0].block10_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 54;
    all_inferences();
    for (int k = 0; k < 64; ++k) {
        par[0].block10_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 11
    stop = 55;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block11_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 56;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block11_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 57;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block11_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 58;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block11_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 59;
    all_inferences();
    for (int k = 0; k < 64; ++k) {
        par[0].block11_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 60;
    all_inferences();
    for (int k = 0; k < 64; ++k) {
        par[0].block11_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }


    // Block 12
    stop = 61;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block12_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 62;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block12_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 63;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block12_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 64;
    all_inferences();
    for (int k = 0; k < 384; ++k) {
        par[0].block12_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 65;
    all_inferences();
    for (int k = 0; k < 96; ++k) {
        par[0].block12_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 66;
    all_inferences();
    for (int k = 0; k < 96; ++k) {
        par[0].block12_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 13
    stop = 67;
    all_inferences();
    for (int k = 0; k < 576; ++k) {
        par[0].block13_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 68;
    all_inferences();
    for (int k = 0; k < 576; ++k) {
        par[0].block13_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 69;
    all_inferences();
    for (int k = 0; k < 576; ++k) {
        par[0].block13_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 70;
    all_inferences();
    for (int k = 0; k < 576; ++k) {
        par[0].block13_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 71;
    all_inferences();
    for (int k = 0; k < 96; ++k) {
        par[0].block13_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 72;
    all_inferences();
    for (int k = 0; k < 96; ++k) {
        par[0].block13_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 14
    stop = 73;
    all_inferences();
    for (int k = 0; k < 576; ++k) {
        par[0].block14_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 74;
    all_inferences();
    for (int k = 0; k < 576; ++k) {
        par[0].block14_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 75;
    all_inferences();
    for (int k = 0; k < 576; ++k) {
        par[0].block14_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 76;
    all_inferences();
    for (int k = 0; k < 576; ++k) {
        par[0].block14_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 77;
    all_inferences();
    for (int k = 0; k < 96; ++k) {
        par[0].block14_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 78;
    all_inferences();
    for (int k = 0; k < 96; ++k) {
        par[0].block14_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 15
    stop = 79;
    all_inferences();
    for (int k = 0; k < 576; ++k) {
        par[0].block15_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 80;
    all_inferences();
    for (int k = 0; k < 576; ++k) {
        par[0].block15_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 81;
    all_inferences();
    for (int k = 0; k < 576; ++k) {
        par[0].block15_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 82;
    all_inferences();
    for (int k = 0; k < 576; ++k) {
        par[0].block15_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 83;
    all_inferences();
    for (int k = 0; k < 160; ++k) {
        par[0].block15_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 84;
    all_inferences();
    for (int k = 0; k < 160; ++k) {
        par[0].block15_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 16
    stop = 85;
    all_inferences();
    for (int k = 0; k < 960; ++k) {
        par[0].block16_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 86;
    all_inferences();
    for (int k = 0; k < 960; ++k) {
        par[0].block16_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 87;
    all_inferences();
    for (int k = 0; k < 960; ++k) {
        par[0].block16_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 88;
    all_inferences();
    for (int k = 0; k < 960; ++k) {
        par[0].block16_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 89;
    all_inferences();
    for (int k = 0; k < 160; ++k) {
        par[0].block16_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 90;
    all_inferences();
    for (int k = 0; k < 160; ++k) {
        par[0].block16_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 17
    stop = 91;
    all_inferences();
    for (int k = 0; k < 960; ++k) {
        par[0].block17_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 92;
    all_inferences();
    for (int k = 0; k < 960; ++k) {
        par[0].block17_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 93;
    all_inferences();
    for (int k = 0; k < 960; ++k) {
        par[0].block17_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 94;
    all_inferences();
    for (int k = 0; k < 960; ++k) {
        par[0].block17_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 95;
    all_inferences();
    for (int k = 0; k < 160; ++k) {
        par[0].block17_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 96;
    all_inferences();
    for (int k = 0; k < 160; ++k) {
        par[0].block17_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Block 18
    stop = 97;
    all_inferences();
    for (int k = 0; k < 960; ++k) {
        par[0].block18_par_expand_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 98;
    all_inferences();
    for (int k = 0; k < 960; ++k) {
        par[0].block18_par_expand_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 99;
    all_inferences();
    for (int k = 0; k < 960; ++k) {
        par[0].block18_par_depth_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 100;
    all_inferences();
    for (int k = 0; k < 960; ++k) {
        par[0].block18_par_depth_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
    stop = 101;
    all_inferences();
    for (int k = 0; k < 320; ++k) {
        par[0].block18_par_project_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 102;
    all_inferences();
    for (int k = 0; k < 320; ++k) {
        par[0].block18_par_project_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }

    // Final Block
    stop = 103;
    all_inferences();
    for (int k = 0; k < 1280; ++k) {
        par[0].initial_par_BN[2][k] = temp_sum[k] / 96;
        temp_sum[k] = 0;
    }
    stop = 104;
    all_inferences();
    for (int k = 0; k < 1280; ++k) {
        par[0].initial_par_BN[3][k] = temp_variance[k] / 96;
        temp_variance[k] = 0;
    }
}

void backprop_fc(float I[1280],
        float O[1000],
        int label,
        float weight[1280][1000],
        float bias[1000],
        float Ew[1280][1000],
        float Eb[1000])
{
    // Allocate gradients
    float (*dLdW)[1000] = calloc(1280, sizeof *dLdW);
    float dLdB[1000];

    for (int j = 0; j < 1000; ++j) {
        for (int i = 0; i < 1280; ++i) {
            // Calculate weight gradient
            if (j == label) {
                dLdW[i][j] = I[i]*(O[j] - 1); //+ LAMBDA*weight[i][j];
            }
            else {
                dLdW[i][j] = I[i]*O[j]; //+ LAMBDA*weight[i][j];
            }
            // Correct weights
            if (epoch_count==0) {Ew[i][j] = 0;}
            Ew[i][j] = E_MOMENTUM*Ew[i][j] + (1 - E_DECAY)*dLdW[i][j]*dLdW[i][j];
            weight[i][j] = weight[i][j] - ( ( (LR*pow(LR_DECAY, epoch_count)) / (sqrt(Ew[i][j])+EPSILON) ) * dLdW[i][j] );
        }
        // Calculate bias gradient
        if (j == label) {
            dLdB[j] = O[j] - 1;
        }
        else {
            dLdB[j] = O[j];
        }
        // Correct biases
        if (epoch_count==0) {Eb[j] = 0;}
        Eb[j] = E_MOMENTUM*Eb[j] + (1 - E_DECAY)*dLdB[j]*dLdB[j];
        bias[j] = bias[j] - ( ( (LR*pow(LR_DECAY, epoch_count)) / (sqrt(Eb[j])+EPSILON) ) * dLdB[j] );
    }
    free(dLdW);
}
