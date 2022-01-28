#ifndef BLOCKS_H
#define BLOCKS_H

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
             float par_project_BN[4][od]);

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
             float par_project_BN[4][pr]);

#endif
