#include "layers.h"
#include "operations.h"

void S2block(int isize, int osize,              // input and output sizes / depthwise factor
             int id, int od,                    // input and output depth / project factor
             int idx2D, int idxBN, int idxDW,   // 2D convolution, Batch Normalization and Depthwise current weight import indexes
             int ex,                            // Expand factor
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
             float par_project_BN[4][od])
{
    // Expand
    conv2d(isize, isize, 1, // input, output, kernel size
            1,              // stride
            0,              // pad input by one pixels de facto centering the kernel
            id, ex,         // number of input and output channels
            idx2D + 1,      // weight import index
            idata, expand, par_expand);

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
            0,              // pad input by one pixels de facto centering the kernel
            ex, od,         // number of input and output channels
            idx2D + 2,      // weight import index
            depth_relu, project, par_project);

    batch_normalization(osize,  // input, output
            od,                 // number of channels
            idxBN + 3,          // weight import index
            project, project_BN, par_project_BN);
}

void S1block(int size,                          // input and output sizes
             int d,                             // input and output depth
             int idx2D, int idxBN, int idxDW,   // 2D convolution, Batch Normalization and Depthwise current weight import indexes
             int ex, int pr, int a,             // Expand, project and add factors
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

    // Expand
    conv2d(size, size, 1,   // input, output, kernel size
            1,              // stride
            0,              // pad input by one pixels de facto centering the kernel
            d, ex,          // number of input and output channels
            idx2D + 1,      // weight import index
            idata, expand, par_expand);

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
            0,              // pad input by one pixels de facto centering the kernel
            ex, pr,         // number of input and output channels
            idx2D + 2,      // weight import index
            depth_relu, project, par_project);

    batch_normalization(size,   // input, output
            pr,                 // number of channels
            idxBN + 3,          // weight import index
            project, add, par_project_BN);

    // Add
    if (a) {
        add_array(size, d, idata, add);
    }
}

void backprop_S1(int size,                          // input and output sizes
                 int d,                             // input and output depth
                 int idx2D, int idxBN, int idxDW,   // 2D convolution, Batch Normalization and Depthwise current weight import indexes
                 int ex, int pr,                    // Expand and project factors
                 float idata[size][size][d],
                 float expand[size][size][ex],
                 float expand_relu[size][size][ex],
                 float depth[size][size][ex],
                 float depth_relu[size][size][ex],
                 float project[size][size][pr],
                 float add[size][size][pr],
                 float par_expand[ex][1][1][d],   float par_expand_E[ex][1][1][d],
                 float par_expand_BN[4][ex],      float par_expand_BN_E[4][ex],
                 float par_depth[3][3][ex],       float par_depth_E[3][3][ex],
                 float par_depth_BN[4][ex],       float par_depth_BN_E[4][ex],
                 float par_project[pr][1][1][ex], float par_project_E[pr][1][1][ex],
                 float par_project_BN[4][pr],     float par_project_BN_E[4][pr])
{
    // Re-Project
    backprop_bn(size, pr,   // Size and Depth
            project, add,
            par_project_BN, par_project_BN_E,
            idxBN + 3);     // Import index

    backprop_conv2d(size, size, 1, ex, pr,  // Line size, column size, kernel size, input depth, output depth
            depth_relu, project,
            par_project, par_project_E,
            1, 0, idx2D + 2);               // Stride, padding and import index

    // Re-Depthwise
    backprop_relu6(size, ex,    // Size and Depth
            depth_relu);

    backprop_bn(size, ex,   // Size and Depth
            depth, depth_relu,
            par_depth_BN, par_depth_BN_E,
            idxBN + 2);     // Import index

    backprop_dw(size, size, 3, ex,  // Input size, output size, kernel size, depth
            expand_relu, depth,
            par_depth, par_depth_E,
            1, 1, idxDW + 1);       // Stride, padding and import index

    // Re-Expand
    backprop_relu6(size, ex,    // Size and Depth
            expand_relu);

    backprop_bn(size, ex,   // Size and Depth
            expand, expand_relu,
            par_expand_BN, par_expand_BN_E,
            idxBN + 1);     // Import index

    backprop_conv2d(size, size, 1, d, ex,   // Line size, column size, kernel size, input depth, output depth
            idata, expand,
            par_expand, par_expand_E,
            1, 0, idx2D + 1);               // Stride, padding and import index
}

void backprop_S2(int isize, int osize,              // input and output sizes / depthwise factor
                 int id, int od,                    // input and output depth / project factor
                 int idx2D, int idxBN, int idxDW,   // 2D convolution, Batch Normalization and Depthwise current weight import indexes
                 int ex,                            // Expand factor
                 float idata[isize][isize][id],
                 float expand[isize][isize][ex],
                 float expand_relu[isize][isize][ex],
                 float depth[osize][osize][ex],
                 float depth_relu[osize][osize][ex],
                 float project[osize][osize][od],
                 float project_BN[osize][osize][od],
                 float par_expand[ex][1][1][id],  float par_expand_E[ex][1][1][id],
                 float par_expand_BN[4][ex],      float par_expand_BN_E[4][ex],
                 float par_depth[3][3][ex],       float par_depth_E[3][3][ex],
                 float par_depth_BN[4][ex],       float par_depth_BN_E[4][ex],
                 float par_project[od][1][1][ex], float par_project_E[od][1][1][ex],
                 float par_project_BN[4][od],     float par_project_BN_E[4][od])
{
    // Re-Project
    backprop_bn(osize, od,  // Size and Depth
            project, project_BN,
            par_project_BN, par_project_BN_E,
            idxBN + 3);     // Import index

    backprop_conv2d(osize, osize, 1, ex, od,    // Line size, column size, kernel size, input depth, output depth
            depth_relu, project,
            par_project, par_project_E,
            1, 0, idx2D + 2);                   // Stride, padding and import index

    // Re-Depthwise
    backprop_relu6(osize, ex,   // Size and Depth
            depth_relu);

    backprop_bn(osize, ex,  // Size and Depth
            depth, depth_relu,
            par_depth_BN, par_depth_BN_E,
            idxBN + 2);     // Import index

    backprop_dw(isize, osize, 3, ex,    // Input size, output size, kernel size, depth
            expand_relu, depth,
            par_depth, par_depth_E,
            2, 0, idxDW + 1);           // Stride, padding and import index

    // Re-Expand
    backprop_relu6(isize, ex,   // Size and Depth
            expand_relu);

    backprop_bn(isize, ex,  // Size and Depth
            expand, expand_relu,
            par_expand_BN, par_expand_BN_E,
            idxBN + 1);     // Import index

    backprop_conv2d(isize, isize, 1, id, ex,    // Line size, column size, kernel size, input depth, output depth
            idata, expand,
            par_expand, par_expand_E,
            1, 0, idx2D + 1);                   // Stride, padding and import index
}
