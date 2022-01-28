#include <stdlib.h>
#include "var.h"
#include "layers.h"
#include "operations.h"

// for testing
#include <stdio.h>

void S2block(int isize, int osize,              // input and output sizes / depthwise factor
             int id, int od,                    // input and output depth / project factor
             int idx2D, int idxBN, int idxDW,   // 2D convolution, Batch Normalization and Depthwise current weight import indexes
             int ex,                            // Expand factor
             int s,                             // Current stop
             float idata[isize][isize][id],
             float expand[isize][isize][ex],
             float expand_relu[isize][isize][ex],
             float depth[osize][osize][ex],
             float depth_relu[osize][osize][ex],
             float project[osize][osize][od],
             float project_BN[osize][osize][od],
             float par_expand[ex][1][1][id],
             float par_expand_BN[4][ex],
             float par_depth[3][3][ex],
             float par_depth_BN[4][ex],
             float par_project[od][1][1][ex],
             float par_project_BN[4][od]) {
    //Expand
    conv2d(isize, isize, 1, // input, output, kernel size
            1,              // stride
            0,              // pad input by one pixels de factor centering the kernel
            id, ex,         // number of input and output channels
            idx2D + 1,      // weight import index
            idata, expand, par_expand);

    if (stop == s) {
        for (int k = 0; k < ex; ++k) {
        for (int i = 0; i < isize; ++i) {
        for (int j = 0; j < isize; ++j) {
            temp_sum[k] += expand[i][j][k];
        }}}
        return;
    }

    if (stop == s+1) {
        for (int k = 0; k < ex; ++k) {
        for (int i = 0; i < isize; ++i) {
        for (int j = 0; j < isize; ++j) {
            temp_variance[k] += (expand[i][j][k] - par_expand_BN[2][k]) * (expand[i][j][k] - par_expand_BN[2][k]);
        }}}
        return;
    }

    batch_normalization(isize,  // input, output
            ex,                 // number of channels
            idxBN + 1,          // weight import index
            expand, expand_relu, par_expand_BN);

    relu6(isize,    // input, output
            ex,     // number of channels
            expand_relu);

    // Depthwise
    S2depthwise(isize, osize, 3,    // input size, output size and kernel size   //isize+1 with padding
            0,                      // padding
            ex,                     // depth
            idxDW + 1,              // weight import index
            expand_relu, depth, par_depth);

    if (stop == s+2) {
        for (int k = 0; k < ex; ++k) {
        for (int i = 0; i < osize; ++i) {
        for (int j = 0; j < osize; ++j) {
            temp_sum[k] += depth[i][j][k];
        }}}
        return;
    }

    if (stop == s+3) {
        for (int k = 0; k < ex; ++k) {
        for (int i = 0; i < osize; ++i) {
        for (int j = 0; j < osize; ++j) {
            temp_variance[k] += (depth[i][j][k] - par_depth_BN[2][k]) * (depth[i][j][k] - par_depth_BN[2][k]);
        }}}
        return;
    }

    batch_normalization(osize,  // input, output
            ex,                 // number of channels
            idxBN + 2,          // weight import index
            depth, depth_relu, par_depth_BN);

    relu6(osize,    // input, output
            ex,     // number of channels
            depth_relu);

    // Project
    conv2d(osize, osize, 1, // input, output, kernel size
            1,              // stride
            0,              // pad input by one pixels de factor centering the kernel
            ex, od,         // number of input and output channels
            idx2D + 2,      // weight import index
            depth_relu, project, par_project);

    if (stop == s+4) {
        for (int k = 0; k < od; ++k) {
        for (int i = 0; i < osize; ++i) {
        for (int j = 0; j < osize; ++j) {
            temp_sum[k] += project[i][j][k];
        }}}
        return;
    }

    if (stop == s+5) {
        for (int k = 0; k < od; ++k) {
        for (int i = 0; i < osize; ++i) {
        for (int j = 0; j < osize; ++j) {
            temp_variance[k] += (project[i][j][k] - par_project_BN[2][k]) * (project[i][j][k] - par_project_BN[2][k]);
        }}}
        return;
    }

    batch_normalization(osize,  // input, output
            od,                 // number of channels
            idxBN + 3,          // weight import index
            project, project_BN, par_project_BN);
}

void S1block(int size,                          // input and output sizes
             int d,                             // input and output depth
             int idx2D, int idxBN, int idxDW,   // 2D convolution, Batch Normalization and Depthwise current weight import indexes
             int ex, int pr, int a,             // Expand, project and add factors
             int s,                             // Current stop
             float idata[size][size][d],
             float expand[size][size][ex],
             float expand_relu[size][size][ex],
             float depth[size][size][ex],
             float depth_relu[size][size][ex],
             float project[size][size][pr],
             float add[size][size][pr],
             float par_expand[ex][1][1][d],
             float par_expand_BN[4][ex],
             float par_depth[3][3][ex],
             float par_depth_BN[4][ex],
             float par_project[pr][1][1][ex],
             float par_project_BN[4][pr]) {

    //Expand
    conv2d(size, size, 1,   // input, output, kernel size
            1,              // stride
            0,              // pad input by one pixels de factor centering the kernel
            d, ex,          // number of input and output channels
            idx2D + 1,      // weight import index
            idata, expand, par_expand);

    if (stop == s) {
        for (int k = 0; k < ex; ++k) {
        for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            temp_sum[k] += expand[i][j][k];
        }}}
        return;
    }

    if (stop == s+1) {
        for (int k = 0; k < ex; ++k) {
        for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            temp_variance[k] += (expand[i][j][k] - par_expand_BN[2][k]) * (expand[i][j][k] - par_expand_BN[2][k]);
        }}}
        return;
    }

    batch_normalization(size,   // input, output
            ex,                 // number of channels
            idxBN + 1,          // weight import index
            expand, expand_relu, par_expand_BN);

    relu6(size, // input, output
            ex, // number of channels
            expand_relu);

    // Depthwise
    S1depthwise(size, 3,    // input size, output size and kernel size   //isize+1 with padding
            1,              // padding
            ex,             // depth
            idxDW + 1,      // weight import index
            expand_relu, depth, par_depth);

    if (stop == s+2) {
        for (int k = 0; k < ex; ++k) {
        for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            temp_sum[k] += depth[i][j][k];
        }}}
        return;
    }

    if (stop == s+3) {
        for (int k = 0; k < ex; ++k) {
        for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            temp_variance[k] += (depth[i][j][k] - par_depth_BN[2][k]) * (depth[i][j][k] - par_depth_BN[2][k]);
        }}}
        return;
    }

    batch_normalization(size,   // input, output
            ex,                 // number of channels
            idxBN + 2,          // weight import index
            depth, depth_relu, par_depth_BN);

    relu6(size, // input, output
            ex, // number of channels
            depth_relu);

    // Project
    conv2d(size, size, 1,   // input, output, kernel size
            1,              // stride
            0,              // pad input by one pixels de factor centering the kernel
            ex, pr,         // number of input and output channels
            idx2D + 2,      // weight import index
            depth_relu, project, par_project);

    if (stop == s+4) {
        for (int k = 0; k < pr; ++k) {
        for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            temp_sum[k] += project[i][j][k];
        }}}
        return;
    }

    if (stop == s+5) {
        for (int k = 0; k < pr; ++k) {
        for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            temp_variance[k] += (project[i][j][k] - par_project_BN[2][k]) * (project[i][j][k] - par_project_BN[2][k]);
        }}}
        return;
    }

    batch_normalization(size,   // input, output
            pr,                 // number of channels
            idxBN + 3,          // weight import index
            project, add, par_project_BN);

    // Add
    if (a) {
        add_array(size, d, idata, add);
    }
}
