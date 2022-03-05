#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "data_manip.h"
#include "var.h"

#define LAMBDA 0.00004
#define EPSILON 0.0000001       // original: 0.0000001
#define E_MOMENTUM 0.9
#define E_DECAY 0.9
#define RHO 0.999

void convolution2D(int isize,   // width/height of input
        int osize,              // width/height of output
        int ksize,              // width/height of kernel
        int stride,             // shift between input pixels, between consecutive outputs
        int pad,                // offset between (0,0) pixels between input and output
        int idepth, int odepth, // number of input and output channels
        float idata[isize][isize][idepth],
        float odata[osize][osize][odepth],
        float kdata[odepth][ksize][ksize][idepth])
{
    // iterate over the output
    for (int oy = 0; oy < osize; ++oy) {
    for (int ox = 0; ox < osize; ++ox) {
    for (int od = 0; od < odepth; ++od) {
        odata[oy][ox][od] = 0;  // When you iterate multiple times without closing the program, this number would stack up to infinity, so we have to zero it out every time.
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

    float (*normI)[size][depth] = calloc(size,sizeof *normI);   // This is so that the actual input is not changed, so that it can be reused in backpropagation.

    for (int d = 0; d < depth; ++d)
    {
        // [0][:] is gamma, [1][:] is beta, [2][:] is moving mean, [3][:] is moving variance
        // Here we use moving mean and variance instead of calculating them, as opposed to training.
        for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            normI[y][x][d] = (idata[y][x][d] - pdata[2][d]) / sqrt(pdata[3][d] + epsilon); // normalize
            odata[y][x][d] = (normI[y][x][d] * pdata[0][d]) + pdata[1][d]; // scale and shift
        }}
    }

    free(normI);
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
        odata[oy][ox][od] = 0;  // When you iterate multiple times without closing the program, this number would stack up to infinity, so we have to zero it out every time.
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

void fully_connect(int isize, int osize, float w[isize][osize][2], float b[osize][2], float *idata, float *odata) {

    for (int y = 0; y < osize; ++y) { // 1000
        odata[y] = 0;
        for (int x = 0; x < isize; ++x) { // 1280
            odata[y] += idata[x]*w[x][y][0];
        }
        odata[y] += b[y][0];
    }
}

struct Correlation {
    float prediction;
    int idx;
};

int compare(const void *p1, const void *p2) {
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

    struct Correlation cor[class];

    for(int n = 0; n < class; n++) {
        cor[n].prediction = pred[n];
        cor[n].idx = n;
    }

    // Acquire correct prediction
    qsort(cor, class, sizeof(struct Correlation), compare);
    inf_idx = cor[0].idx;
    inf_pred = cor[0].prediction;
/*
    FILE *fp;
    fp = fopen("fc_pred.csv", "w");
    if (fp == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "%.7e\n%d\n", cor[0].prediction, cor[0].idx);
    fclose(fp);
    printf("Saved fc_pred.csv\n");
*/

    // Print correlations
    int n_cor;
    if (class >= 5)
        n_cor = 5;
    else
        n_cor = class;

    printf("\n");
    for (int n = 0; n < n_cor; n++)
        printf("Correlation %d: %e / Index = %d\n", n+1, cor[n].prediction, cor[n].idx);
    printf("\n");

    // Print label correlations
    FILE *fptr;
    fptr = fopen("labels.txt", "r");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    char s[122];
    int j;

    for (int n = 0; n < n_cor; n++)
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

void backprop_fc(int c,
        float I[1280],
        float O[c],
        int label,
        float fc_w[1280][c][2],
        float fc_b[c][2])
{

    float (*dLdW)[c] = calloc(1280, sizeof *dLdW);   // We need this because both the backprop error and the weight gradient calculations depend on W and I
    float dLdB[c];

    // Calculate gradient and moving squared means - uses input
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < 1280; ++i) {
            if (j == label) {
                dLdW[i][j] = I[i]*(O[j] - 1) + LAMBDA*fc_w[i][j][0];
            }
            else {
                dLdW[i][j] = I[i]*O[j] + LAMBDA*fc_w[i][j][0];
            }

            fc_w[i][j][1] = E_MOMENTUM*fc_w[i][j][1] + (1 - E_DECAY)*dLdW[i][j]*dLdW[i][j];
        }
        if (j == label) {
            dLdB[j] = O[j] - 1;
        }
        else {
            dLdB[j] = O[j];
        }

        fc_b[j][1] = E_MOMENTUM*fc_b[j][1] + (1 - E_DECAY)*dLdB[j]*dLdB[j];
    }

    if (frz != 1) {

        // Calculate backpropagated error (saved in final_pooling[1280] for memory's sake) - uses weights, changes inputs
        for (int i = 0; i < 1280; ++i) {
            I[i] = 0;
            for (int j = 0; j < c; ++j) {
                if (j == label) {
                    I[i] += fc_w[i][j][0] * (O[j] - 1);
                }
                else {
                    I[i] += fc_w[i][j][0] * O[j];
                }
            }
        }

    }

    // Correct weights and biases - changes weights
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < 1280; ++i) {
            fc_w[i][j][0] = fc_w[i][j][0] - ( ( (lr*pow(lr_decay, epoch_count)) / (sqrt(fc_w[i][j][1])+EPSILON) ) * dLdW[i][j] );
        }
    fc_b[j][0] = fc_b[j][0] - ( ( (lr*pow(lr_decay, epoch_count)) / (sqrt(fc_b[j][1])+EPSILON) ) * dLdB[j] );
    }
    free(dLdW);

    // For analyzing overfitting
    //printf("fc_w convergence index: %.7e\n\n", fabs( fabs(fc_w[0][0][0]) - fabs(fc_w[0][1][0]) ) );

    export_fc(1280, c, fc_w, fc_b);
}

void backprop_avrgpool(float I[7][7][1280], float O[1280]) {

    for (int k = 0; k < 1280; ++k) {
        float temp = O[k] / 49;
        for (int j = 0; j < 7; ++j) {
            for (int i = 0; i < 7; ++i) {
                I[i][j][k] = temp;
            }
        }
    }

}

void backprop_relu6(int s, int d, float I[s][s][d]) {

    for (int i = 0; i < s; ++i) {
        for (int j = 0; j < s; ++j) {
            for (int k = 0; k < d; ++k) {
                if (I[i][j][k] < (float) 0 || I[i][j][k] > (float) 6) {
                    I[i][j][k] = 0;
                }
            }
        }
    }

}

void backprop_bn(int s, int d,
        float I[s][s][d], float O[s][s][d],
        float par[4][d], float Ep[4][d],
        int idx)
{
    // [0][:] is gamma, [1][:] is beta, [2][:] is moving mean, [3][:] is moving variance

    float Bgrad;
    float Ihat;

    // Correct Beta - Beta is not used in the backpropagation step
    for (int k = 0; k < d; ++k) {
        Bgrad = 0;
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                Bgrad += O[i][j][k];
            }
        }
        Ep[1][k] = E_MOMENTUM*Ep[1][k] + (1 - E_DECAY)*Bgrad*Bgrad;
        par[1][k] = par[1][k] - ( ( (lr*pow(lr_decay, epoch_count)) / (sqrt(Ep[1][k])+EPSILON) ) * Bgrad );
    }

    // Calculate Gamma Gradient
    float Ggrad[d];
    for (int k = 0; k < d; ++k) {
        Ggrad[k] = 0;
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                Ihat = (I[i][j][k] - par[2][k]) / sqrt(par[3][k] + EPSILON);
                Ggrad[k] += O[i][j][k] * Ihat;
            }
        }
    }

    // Backpropagate
    float sum;
    float dMdI = 1.0 / (s*s);                           // Calc dM/dI
    for (int k = 0; k < d; ++k) {
        float dIhatdI = 1/sqrt(par[3][k] + EPSILON);    // Calc dIhat/dI
        float dVdI[s][s];
        float dLdIhat[s][s];
        float dLdM = 0;
        float dLdV = 0;

        // Variables
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                dVdI[i][j] = (2*(I[i][j][k] - par[2][k])) / (s*s);                                                          // Calc dV/dI
                dLdIhat[i][j] = O[i][j][k] * par[0][k];                                                                     // Calc dL/dIhat
                dLdM += dLdIhat[i][j] * ( -1 / sqrt(par[3][k] + EPSILON) );                                                 // Calc dL/dM
                dLdV += dLdIhat[i][j] * (I[i][j][k] - par[2][k]) * (-1.0/2) * pow(par[3][k] + EPSILON, -3/2);               // Calc dL/dV
            }
        }

        // New Moving Mean and Moving Variance
        sum = 0;
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                sum += I[i][j][k];
            }
        }
        par[2][k] = par[2][k] * RHO + (1 - RHO) * (sum / (s*s));

        sum = 0;
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                sum += (I[i][j][k] - par[2][k]) * (I[i][j][k] - par[2][k]);
            }
        }
        par[3][k] = par[3][k] * RHO + (1 - RHO) * (sum / (s*s));

        // Error
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                I[i][j][k] = (dLdIhat[i][j] * dIhatdI) + (dLdV * dVdI[i][j]) + (dLdM * dMdI);   // Calc Error based on previous 6 variables and update

                // FOR DEBUGGING
                if ( (i == 0) & (j == 0) & (k == 0) )
                    printf("\ndLdIhat = %.7e   dIhatdI = %.7e   dLdV = %.7e   dVdI = %.7e   dLdM = %.7e   dMdI = %.7e\n\n", dLdIhat[i][j], dIhatdI, dLdV, dVdI[i][j], dLdM, dMdI);
            }
        }


    }

    // Correct Gamma
    for (int k = 0; k < d; ++k) {
        Ep[0][k] = E_MOMENTUM*Ep[0][k] + (1 - E_DECAY)*Ggrad[k]*Ggrad[k];
        par[0][k] = par[0][k] - ( ( (lr*pow(lr_decay, epoch_count)) / (sqrt(Ep[0][k])+EPSILON) ) * Ggrad[k] );
    }


/*              Can't calculate things here because 'I' is now backpropagated error
    // Calculate new Moving Mean and Moving Variance
    float sum;
    for (int k = 0; k < d; ++k) {
        sum = 0;
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                sum += I[i][j][k];
            }
        }
        par[2][k] = par[2][k] * RHO + (1 - RHO) * (sum / (s*s));

        sum = 0;
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                sum += (I[i][j][k] - par[2][k]) * (I[i][j][k] - par[2][k]);
            }
        }
        par[3][k] = par[3][k] * RHO + (1 - RHO) * (sum / (s*s));
    }
*/

    // Export
    char name[12]; // maximum number of characters is "paramxx.csv" = 11
    sprintf(name, "param%d.csv", idx);
    export_bn(name, d, par);
}

void backprop_conv2d(int isize, int osize, int ksize, int idepth, int odepth,
        float I[isize][isize][idepth], float O[osize][osize][odepth],
        float par[odepth][ksize][ksize][idepth], float Ew[odepth][ksize][ksize][idepth],
        int stride, int pad, int idx)
{
    int backprop = 1;
    if ( (frz == 2  && idx == 35) ||   // Freezing Block
         (frz == 3  && idx == 33) ||
         (frz == 4  && idx == 31) ||
         (frz == 5  && idx == 29) ||
         (frz == 6  && idx == 27) ||
         (frz == 7  && idx == 25) ||
         (frz == 8  && idx == 23) ||
         (frz == 9  && idx == 21) ||
         (frz == 10 && idx == 19) ||
         (frz == 11 && idx == 17) ||
         (frz == 12 && idx == 15) ||
         (frz == 13 && idx == 13) ||
         (frz == 14 && idx == 11) ||
         (frz == 15 && idx == 9)  ||
         (frz == 16 && idx == 7)  ||
         (frz == 17 && idx == 5)  ||
         (frz == 18 && idx == 3) )
        backprop = 0;

    float (*dLdW)[ksize][ksize][idepth] = calloc(odepth, sizeof *dLdW);
    float (*error)[isize][idepth] = calloc(isize, sizeof *error);


    // iterate over the output
    for (int oy = 0; oy < osize; ++oy) { // p
    for (int ox = 0; ox < osize; ++ox) { // q
    for (int od = 0; od < odepth; ++od) { // k
        for (int ky = 0; ky < ksize; ++ky) { // i
        for (int kx = 0; kx < ksize; ++kx) { // j
            // map position in output and kernel to the input
            int iy = stride * oy + ky - pad;
            int ix = stride * ox + kx - pad;
            // use only valid inputs
            if (iy >= 0 && iy < isize && ix >= 0 && ix < isize)
                for (int id = 0; id < idepth; ++id) { // c
                    dLdW[od][ky][kx][id] += O[oy][ox][od] * I[iy][ix][id];          // Calculate Gradient
                    if (backprop != 0)
                        error[iy][ix][id] += O[oy][ox][od] * par[od][ky][kx][id];   // Backpropagate
                }
        }}
    }}}

    if (backprop != 0) {
        // I receives Error
        for (int i = 0; i < isize; ++i) {
        for (int j = 0; j < isize; ++j) {
        for (int k = 0; k < idepth; ++k) {
            I[i][j][k] = error[i][j][k];
        }}}
    }

    free(error);

    // Update weights
    for (int od = 0; od < odepth; ++od) {
    for (int ky = 0; ky < ksize; ++ky) {
    for (int kx = 0; kx < ksize; ++kx) {
    for (int id = 0; id < idepth; ++id){
        Ew[od][ky][kx][id] = E_MOMENTUM*Ew[od][ky][kx][id] + (1 - E_DECAY)*dLdW[od][ky][kx][id]*dLdW[od][ky][kx][id];
        par[od][ky][kx][id] = par[od][ky][kx][id] - ( ( (lr*pow(lr_decay, epoch_count)) / (sqrt(Ew[od][ky][kx][id])+EPSILON) ) * dLdW[od][ky][kx][id] );
    }}}}
    free(dLdW);

    // Export
    char name[14]; // maximum number of characters is "weightsxx.csv" = 13
    sprintf(name, "weights%d.csv", idx);
    export_conv2d(name,odepth,ksize,ksize,idepth,par);
}

void backprop_dw(int isize, int osize, int ksize, int depth,
        float I[isize][isize][depth], float O[osize][osize][depth],
        float par[ksize][ksize][depth], float Ew[ksize][ksize][depth],
        int stride, int pad, int idx)
{

    float (*dLdW)[ksize][depth] = calloc(ksize, sizeof *dLdW);

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
                dLdW[ky][kx][od] += O[oy][ox][od] * I[iy][ix][od];      // Calculate Gradient
        }}
    }}}

    // Zero I
    for (int i = 0; i < isize; ++i) {
    for (int j = 0; j < isize; ++j) {
    for (int k = 0; k < depth; ++k) {
        I[i][j][k] = 0;
    }}}

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
                I[iy][ix][od] += O[oy][ox][od] * par[ky][kx][od];        // Backpropagate
        }}
    }}}

    // Update weights
    for (int ky = 0; ky < ksize; ++ky) {
    for (int kx = 0; kx < ksize; ++kx) {
    for (int od = 0; od < depth; ++od){
        Ew[ky][kx][od] = E_MOMENTUM*Ew[ky][kx][od] + (1 - E_DECAY)*dLdW[ky][kx][od]*dLdW[ky][kx][od];
        par[ky][kx][od] = par[ky][kx][od] - ( ( (lr*pow(lr_decay, epoch_count)) / (sqrt(Ew[ky][kx][od])+EPSILON) ) * dLdW[ky][kx][od] );
    }}}
    free(dLdW);

    // Export
    char name[15]; // maximum number of characters is "dweightsxx.csv" = 14
    sprintf(name, "dweights%d.csv", idx);
    export_depth(name, ksize, depth, par);
}
