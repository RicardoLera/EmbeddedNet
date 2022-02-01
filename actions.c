#include <stdlib.h>
#include <stdio.h>
#include "var.h"
#include "blocks.h"
#include "layers.h"
#include "operations.h"
#include "data_manip.h"

// Initial Block
float initial_conv2d[112][112][32];
float initial_relu[112][112][32];

// Expanded Block
float expanded_depth[112][112][32];
float expanded_relu[112][112][32];
float expanded_project[112][112][16];
float expanded_project_BN[112][112][16];

// Block 3 (Stride 2)
float block3_expand[112][112][96];
float block3_expand_relu[112][112][96];
float block3_depth[56][56][96];
float block3_depth_relu[56][56][96];
float block3_project[56][56][24];
float block3_project_BN[56][56][24];

// Block 4 (Stride 1)
float block4_expand[56][56][144];
float block4_expand_relu[56][56][144];
float block4_depth[56][56][144];
float block4_depth_relu[56][56][144];
float block4_project[56][56][24];
float block4_add[56][56][24];

// Block 5 (Stride 2)
float block5_expand[56][56][144];
float block5_expand_relu[56][56][144];
float block5_depth[28][28][144];
float block5_depth_relu[28][28][144];
float block5_project[28][28][32];
float block5_project_BN[28][28][32];

// Block 6 (Stride 1)
float block6_expand[28][28][192];
float block6_expand_relu[28][28][192];
float block6_depth[28][28][192];
float block6_depth_relu[28][28][192];
float block6_project[28][28][32];
float block6_add[28][28][32];

// Block 7 (Stride 1)
float block7_expand[28][28][192];
float block7_expand_relu[28][28][192];
float block7_depth[28][28][192];
float block7_depth_relu[28][28][192];
float block7_project[28][28][32];
float block7_add[28][28][32];

// Block 8 (Stride 2)
float block8_expand[28][28][192];
float block8_expand_relu[28][28][192];
float block8_depth[14][14][192];
float block8_depth_relu[14][14][192];
float block8_project[14][14][64];
float block8_project_BN[14][14][64];

// Block 9 (Stride 1)
float block9_expand[14][14][384];
float block9_expand_relu[14][14][384];
float block9_depth[14][14][384];
float block9_depth_relu[14][14][384];
float block9_project[14][14][64];
float block9_add[14][14][64];

// Block 10 (Stride 1)
float block10_expand[14][14][384];
float block10_expand_relu[14][14][384];
float block10_depth[14][14][384];
float block10_depth_relu[14][14][384];
float block10_project[14][14][64];
float block10_add[14][14][64];

// Block 11 (Stride 1)
float block11_expand[14][14][384];
float block11_expand_relu[14][14][384];
float block11_depth[14][14][384];
float block11_depth_relu[14][14][384];
float block11_project[14][14][64];
float block11_add[14][14][64];

// Block 12 (Stride 1)
float block12_expand[14][14][384];
float block12_expand_relu[14][14][384];
float block12_depth[14][14][384];
float block12_depth_relu[14][14][384];
float block12_project[14][14][96];
float block12_add[14][14][96];

// Block 13 (Stride 1)
float block13_expand[14][14][576];
float block13_expand_relu[14][14][576];
float block13_depth[14][14][576];
float block13_depth_relu[14][14][576];
float block13_project[14][14][96];
float block13_add[14][14][96];

// Block 14 (Stride 1)
float block14_expand[14][14][576];
float block14_expand_relu[14][14][576];
float block14_depth[14][14][576];
float block14_depth_relu[14][14][576];
float block14_project[14][14][96];
float block14_add[14][14][96];

// Block 15 (Stride 2)
float block15_expand[14][14][576];
float block15_expand_relu[14][14][576];
float block15_depth[7][7][576];
float block15_depth_relu[7][7][576];
float block15_project[7][7][160];
float block15_project_BN[7][7][160];

// Block 16 (Stride 1)
float block16_expand[7][7][960];
float block16_expand_relu[7][7][960];
float block16_depth[7][7][960];
float block16_depth_relu[7][7][960];
float block16_project[7][7][160];
float block16_add[7][7][160];

// Block 17 (Stride 1)
float block17_expand[7][7][960];
float block17_expand_relu[7][7][960];
float block17_depth[7][7][960];
float block17_depth_relu[7][7][960];
float block17_project[7][7][160];
float block17_add[7][7][160];

// Block 18 (Stride 1)
float block18_expand[7][7][960];
float block18_expand_relu[7][7][960];
float block18_depth[7][7][960];
float block18_depth_relu[7][7][960];
float block18_project[7][7][320];
float block18_add[7][7][320];

// Final Block
float final_conv2d[7][7][1280];
float final_conv2d_relu[7][7][1280];
float final_pooling[1280];
float predictions[1000];


void inference() {

    // Initial Block
    printf("Entering Initial Block: 224x224x3\n");

    conv2d(224, 112, 3, // input, output, kernel size
            2,          // stride
            0,          // pad input by one pixels de factor centering the kernel
            3, 32,      // number of input and output channels
            1,          // weight import index
            image, initial_conv2d, par[0].initial_par_conv2d);

    batch_normalization(112,    // input, output
            32,                 // number of channels
            1,                  // weight import index
            initial_conv2d, initial_relu, par[0].initial_par_BN);

    relu6(112,  // input, output
            32, // number of channels
            initial_relu);  // This one goes straight to "operations" because there is no parameter allocation necessary


    // Expanded Block
    printf("Entering Expanded Block: 112x112x32\n");

    S1depthwise(112, 3, // size and kernel size
            1,          // padding
            32,         // depth
            1,          // weight import index
            initial_relu, expanded_depth, par[0].expanded_par_depth);

    batch_normalization(112,    // input, output
            32,                 // number of channels
            2,                  // weight import index
            expanded_depth, expanded_relu, par[0].expanded_par_depth_BN);

    relu6(112,  // input, output
            32, // number of channels
            expanded_relu);

    conv2d(112, 112, 1, // input, output, kernel size
            1,          // stride
            0,          // pad input by one pixels de factor centering the kernel
            32, 16,     // number of input and output channels
            2,          // weight import index
            expanded_relu, expanded_project, par[0].expanded_par_project);

    batch_normalization(112,    // input, output
            16,                 // number of channels
            3,                  // weight import index
            expanded_project, expanded_project_BN, par[0].expanded_par_project_BN);


    // Block 3 - Stride 2 (Block 1 in Python)
    printf("Entering Block 3: 112x112x16\n");

    S2block(112, 56,    // input and output sizes / depthwise factor
            16, 24,     // input and output depth / project factor
            2, 3, 1,    // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            96,         // Expand factor
            expanded_project_BN,
            block3_expand, block3_expand_relu,
            block3_depth, block3_depth_relu,
            block3_project, block3_project_BN,
            par[0].block3_par_expand, par[0].block3_par_expand_BN,
            par[0].block3_par_depth, par[0].block3_par_depth_BN,
            par[0].block3_par_project, par[0].block3_par_project_BN);


    // Block 4 - Stride 1 (Block 2 in Python)
    printf("Entering Block 4: 56x56x24\n");

    S1block(56,         // input and output sizes
            24,         // input depth
            4, 6, 2,    // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            144, 24, 1, // Expand, project and add factors
            block3_project_BN,
            block4_expand, block4_expand_relu,
            block4_depth, block4_depth_relu,
            block4_project, block4_add,
            par[0].block4_par_expand, par[0].block4_par_expand_BN,
            par[0].block4_par_depth, par[0].block4_par_depth_BN,
            par[0].block4_par_project, par[0].block4_par_project_BN);


    // Block 5 - Stride 2 (Block 3 in Python)
    printf("Entering Block 5: 56x56x24\n");

    S2block(56, 28,     // input and output sizes / depthwise factor
            24, 32,     // input and output depth / project factor
            6, 9, 3,    // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            144,        // Expand factor
            block4_add,
            block5_expand, block5_expand_relu,
            block5_depth, block5_depth_relu,
            block5_project, block5_project_BN,
            par[0].block5_par_expand, par[0].block5_par_expand_BN,
            par[0].block5_par_depth, par[0].block5_par_depth_BN,
            par[0].block5_par_project, par[0].block5_par_project_BN);


    // Block 6 - Stride 1 (Block 4 in Python)
    printf("Entering Block 6: 28x28x32\n");

    S1block(28,         // input and output sizes
            32,         // input depth
            8, 12, 4,   // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            192, 32, 1, // Expand, project and add factors
            block5_project_BN,
            block6_expand, block6_expand_relu,
            block6_depth, block6_depth_relu,
            block6_project, block6_add,
            par[0].block6_par_expand, par[0].block6_par_expand_BN,
            par[0].block6_par_depth, par[0].block6_par_depth_BN,
            par[0].block6_par_project, par[0].block6_par_project_BN);


    // Block 7 - Stride 1 (Block 5 in Python)
    printf("Entering Block 7: 28x28x32\n");

    S1block(28,         // input and output sizes
            32,         // input depth
            10, 15, 5,  // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            192, 32, 1, // Expand, project and add factors
            block6_add,
            block7_expand, block7_expand_relu,
            block7_depth, block7_depth_relu,
            block7_project, block7_add,
            par[0].block7_par_expand, par[0].block7_par_expand_BN,
            par[0].block7_par_depth, par[0].block7_par_depth_BN,
            par[0].block7_par_project, par[0].block7_par_project_BN);


    // Block 8 - Stride 2 (Block 6 in Python)
    printf("Entering Block 8: 28x28x32\n");

    S2block(28, 14,     // input and output sizes / depthwise factor
            32, 64,     // input and output depth / project factor
            12, 18, 6,  // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            192,        // Expand factor
            block7_add,
            block8_expand, block8_expand_relu,
            block8_depth, block8_depth_relu,
            block8_project, block8_project_BN,
            par[0].block8_par_expand, par[0].block8_par_expand_BN,
            par[0].block8_par_depth, par[0].block8_par_depth_BN,
            par[0].block8_par_project, par[0].block8_par_project_BN);


    // Block 9 - Stride 1 (Block 7 in Python)
    printf("Entering Block 9: 14x14x64\n");

    S1block(14,         // input and output sizes
            64,         // input depth
            14, 21, 7,  // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 64, 1, // Expand, project and add factors
            block8_project_BN,
            block9_expand, block9_expand_relu,
            block9_depth, block9_depth_relu,
            block9_project, block9_add,
            par[0].block9_par_expand, par[0].block9_par_expand_BN,
            par[0].block9_par_depth, par[0].block9_par_depth_BN,
            par[0].block9_par_project, par[0].block9_par_project_BN);


    // Block 10 - Stride 1 (Block 8 in Python)
    printf("Entering Block 10: 14x14x64\n");

    S1block(14,         // input and output sizes
            64,         // input depth
            16, 24, 8,  // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 64, 1, // Expand, project and add factors
            block9_add,
            block10_expand, block10_expand_relu,
            block10_depth, block10_depth_relu,
            block10_project, block10_add,
            par[0].block10_par_expand, par[0].block10_par_expand_BN,
            par[0].block10_par_depth, par[0].block10_par_depth_BN,
            par[0].block10_par_project, par[0].block10_par_project_BN);


    // Block 11 - Stride 1 (Block 9 in Python)
    printf("Entering Block 11: 14x14x64\n");

    S1block(14,         // input and output sizes
            64,         // input depth
            18, 27, 9,  // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 64, 1, // Expand, project and add factors
            block10_add,
            block11_expand, block11_expand_relu,
            block11_depth, block11_depth_relu,
            block11_project, block11_add,
            par[0].block11_par_expand, par[0].block11_par_expand_BN,
            par[0].block11_par_depth, par[0].block11_par_depth_BN,
            par[0].block11_par_project, par[0].block11_par_project_BN);


    // Block 12 - Stride 1 - Over-Project (Block 10 in Python)
    printf("Entering Block 12: 14x14x64\n");

    S1block(14,         // input and output sizes
            64,         // input depth
            20, 30, 10, // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 96, 0, // Expand, project and add factors
            block11_add,
            block12_expand, block12_expand_relu,
            block12_depth, block12_depth_relu,
            block12_project, block12_add,
            par[0].block12_par_expand, par[0].block12_par_expand_BN,
            par[0].block12_par_depth, par[0].block12_par_depth_BN,
            par[0].block12_par_project, par[0].block12_par_project_BN);


    // Block 13 - Stride 1 (Block 11 in Python)
    printf("Entering Block 13: 14x14x96\n");

    S1block(14,         // input and output sizes
            96,         // input depth
            22, 33, 11, // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            576, 96, 1, // Expand, project and add factors
            block12_add,
            block13_expand, block13_expand_relu,
            block13_depth, block13_depth_relu,
            block13_project, block13_add,
            par[0].block13_par_expand, par[0].block13_par_expand_BN,
            par[0].block13_par_depth, par[0].block13_par_depth_BN,
            par[0].block13_par_project, par[0].block13_par_project_BN);


    // Block 14 - Stride 1 (Block 12 in Python)
    printf("Entering Block 14: 14x14x96\n");

    S1block(14,         // input and output sizes
            96,         // input depth
            24, 36, 12, // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            576, 96, 1, // Expand, project and add factors
            block13_add,
            block14_expand, block14_expand_relu,
            block14_depth, block14_depth_relu,
            block14_project, block14_add,
            par[0].block14_par_expand, par[0].block14_par_expand_BN,
            par[0].block14_par_depth, par[0].block14_par_depth_BN,
            par[0].block14_par_project, par[0].block14_par_project_BN);


    // Block 15 - Stride 2 (Block 13 in Python)
    printf("Entering Block 15: 14x14x96\n");

    S2block(14, 7,      // input and output sizes / depthwise factor
            96, 160,    // input depth / project factor
            26, 39, 13, // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            576,        // Expand factor
            block14_add,
            block15_expand, block15_expand_relu,
            block15_depth, block15_depth_relu,
            block15_project, block15_project_BN,
            par[0].block15_par_expand, par[0].block15_par_expand_BN,
            par[0].block15_par_depth, par[0].block15_par_depth_BN,
            par[0].block15_par_project, par[0].block15_par_project_BN);


    // Block 16 - Stride 1 (Block 14 in Python)
    printf("Entering Block 16: 7x7x160\n");

    S1block(7,              // input and output sizes
            160,            // input depth
            28, 42, 14,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            960, 160, 1,    // Expand, project and add factors
            block15_project_BN,
            block16_expand, block16_expand_relu,
            block16_depth, block16_depth_relu,
            block16_project, block16_add,
            par[0].block16_par_expand, par[0].block16_par_expand_BN,
            par[0].block16_par_depth, par[0].block16_par_depth_BN,
            par[0].block16_par_project, par[0].block16_par_project_BN);


    // Block 17 - Stride 1 (Block 15 in Python)
    printf("Entering Block 17: 7x7x160\n");

    S1block(7,              // input and output sizes
            160,            // input depth
            30, 45, 15,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            960, 160, 1,    // Expand, project and add factors
            block16_add,
            block17_expand, block17_expand_relu,
            block17_depth, block17_depth_relu,
            block17_project, block17_add,
            par[0].block17_par_expand, par[0].block17_par_expand_BN,
            par[0].block17_par_depth, par[0].block17_par_depth_BN,
            par[0].block17_par_project, par[0].block17_par_project_BN);


    // Block 18 - Stride 1 - Over-Project (Block 16 in Python)
    printf("Entering Block 18: 7x7x160\n");

    S1block(7,              // input and output sizes
            160,            // input depth
            32, 48, 16,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            960, 320, 0,    // Expand, project and add factors
            block17_add,
            block18_expand, block18_expand_relu,
            block18_depth, block18_depth_relu,
            block18_project, block18_add,
            par[0].block18_par_expand, par[0].block18_par_expand_BN,
            par[0].block18_par_depth, par[0].block18_par_depth_BN,
            par[0].block18_par_project, par[0].block18_par_project_BN);


    // Final Block
    printf("Entering Final Block: 7x7x320\n");

    conv2d(7, 7, 1,     // input, output, kernel size
            1,          // stride
            0,          // pad input by one pixels de factor centering the kernel
            320, 1280,  // number of input and output channels
            35,         // weight import index
            block18_add, final_conv2d, par[0].final_par_conv2d);

    batch_normalization(7,  // input, output
            1280,           // number of channels
            52,             // weight import index
            final_conv2d, final_conv2d_relu, par[0].final_par_conv2d_BN);

    relu6(7,        // input, output
            1280,   // number of channels
            final_conv2d_relu);

    avgpool(final_conv2d_relu, final_pooling);

    fully_connected(final_pooling, predictions, par[0].final_par_fc_w, par[0].final_par_fc_b);

    softmax(predictions, 1000);

    decode(predictions);

/*

    // Pred Probe Test

    FILE *fp;
    fp = fopen("output.txt", "w");
    for (int i = 0; i < 1000; ++i) {
        fprintf(fp, "%e\n", pred[i]);
    }
    fclose(fp);
    printf("Saved output of layer as \"output.txt\"\n");

*/
/*

    // Probe Test

    printf("output[0][0][0] = %.7f\n", data20[0][0][0]);

    FILE *fp;
    fp = fopen("output.txt", "w");
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            fprintf(fp, "%.7e, ", data20[i][j][0]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Saved output of layer as \"output.txt\"\n");

*/

}

void train(){

    inference();
    printf("\n");

    // Final Block
    backprop_fc(final_pooling,  // Input of layer
            predictions,        // Output of layer
            label,
            par[0].final_par_fc_w, par[0].final_par_fc_b,
            par[1].final_par_fc_w, par[1].final_par_fc_b);

    backprop_avrgpool(final_conv2d_relu,    // Input of layer
            final_pooling                   // Output of layer (now backpropagated error)
            );

    backprop_relu6(7, 1280,     // Dimensions
            final_conv2d_relu   // Input and Output (error)
            );

    backprop_bn(7, 1280,
            final_conv2d, final_conv2d_relu,
            par[0].final_par_conv2d_BN, par[1].final_par_conv2d_BN,
            52);

    backprop_conv2d(7, 7, 1, 320, 1280,
            block18_add, final_conv2d,
            par[0].final_par_conv2d, par[1].final_par_conv2d,
            1, 0, 35);

    // Block 18
    backprop_S1(7,          // input and output sizes
            160,            // input depth
            32, 48, 16,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            960, 320,       // Expand and project factors
            block17_add,
            block18_expand, block18_expand_relu,
            block18_depth, block18_depth_relu,
            block18_project, block18_add,
            par[0].block18_par_expand, par[1].block18_par_expand,
            par[0].block18_par_expand_BN, par[1].block18_par_expand_BN,
            par[0].block18_par_depth, par[1].block18_par_depth,
            par[0].block18_par_depth_BN, par[1].block18_par_depth_BN,
            par[0].block18_par_project, par[1].block18_par_project,
            par[0].block18_par_project_BN, par[1].block18_par_project_BN);

    // Block 17
    backprop_S1(7,          // input and output sizes
            160,            // input depth
            30, 45, 15,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            960, 160,       // Expand and project factors
            block16_add,
            block17_expand, block17_expand_relu,
            block17_depth, block17_depth_relu,
            block17_project, block17_add,
            par[0].block17_par_expand, par[1].block17_par_expand,
            par[0].block17_par_expand_BN, par[1].block17_par_expand_BN,
            par[0].block17_par_depth, par[1].block17_par_depth,
            par[0].block17_par_depth_BN, par[1].block17_par_depth_BN,
            par[0].block17_par_project, par[1].block17_par_project,
            par[0].block17_par_project_BN, par[1].block17_par_project_BN);

    // Block 16
    backprop_S1(7,          // input and output sizes
            160,            // input depth
            28, 42, 14,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            960, 160,       // Expand and project factors
            block15_project_BN,
            block16_expand, block16_expand_relu,
            block16_depth, block16_depth_relu,
            block16_project, block16_add,
            par[0].block16_par_expand, par[1].block16_par_expand,
            par[0].block16_par_expand_BN, par[1].block16_par_expand_BN,
            par[0].block16_par_depth, par[1].block16_par_depth,
            par[0].block16_par_depth_BN, par[1].block16_par_depth_BN,
            par[0].block16_par_project, par[1].block16_par_project,
            par[0].block16_par_project_BN, par[1].block16_par_project_BN);

    // Block 15
    backprop_S2(14, 7,      // input and output sizes / depthwise factor
            96, 160,        // input and output depth / project factor
            26, 39, 13,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            576,            // Expand factor
            block14_add,
            block15_expand, block15_expand_relu,
            block15_depth, block15_depth_relu,
            block15_project, block15_project_BN,
            par[0].block15_par_expand, par[1].block15_par_expand,
            par[0].block15_par_expand_BN, par[1].block15_par_expand_BN,
            par[0].block15_par_depth, par[1].block15_par_depth,
            par[0].block15_par_depth_BN, par[1].block15_par_depth_BN,
            par[0].block15_par_project, par[1].block15_par_project,
            par[0].block15_par_project_BN, par[1].block15_par_project_BN);


/* Zero Backprop Test - Showed that the bug is NOT in the backpropagation
    for (int i = 0; i < 14; ++i) {
    for (int j = 0; j < 14; ++j) {
    for (int k = 0; k < 96; ++k) {
        block14_add[i][j][k] = 0;
    }}}
*/


    // Block 14
    backprop_S1(14,         // input and output sizes
            96,             // input depth
            24, 36, 12,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            576, 96,        // Expand and project factors
            block13_add,
            block14_expand, block14_expand_relu,
            block14_depth, block14_depth_relu,
            block14_project, block14_add,
            par[0].block14_par_expand, par[1].block14_par_expand,
            par[0].block14_par_expand_BN, par[1].block14_par_expand_BN,
            par[0].block14_par_depth, par[1].block14_par_depth,
            par[0].block14_par_depth_BN, par[1].block14_par_depth_BN,
            par[0].block14_par_project, par[1].block14_par_project,
            par[0].block14_par_project_BN, par[1].block14_par_project_BN);

    // Block 13
    backprop_S1(14,         // input and output sizes
            96,             // input depth
            22, 33, 11,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            576, 96,        // Expand and project factors
            block12_add,
            block13_expand, block13_expand_relu,
            block13_depth, block13_depth_relu,
            block13_project, block13_add,
            par[0].block13_par_expand, par[1].block13_par_expand,
            par[0].block13_par_expand_BN, par[1].block13_par_expand_BN,
            par[0].block13_par_depth, par[1].block13_par_depth,
            par[0].block13_par_depth_BN, par[1].block13_par_depth_BN,
            par[0].block13_par_project, par[1].block13_par_project,
            par[0].block13_par_project_BN, par[1].block13_par_project_BN);

    // Block 12
    backprop_S1(14,         // input and output sizes
            64,             // input depth
            20, 30, 10,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 96,        // Expand and project factors
            block11_add,
            block12_expand, block12_expand_relu,
            block12_depth, block12_depth_relu,
            block12_project, block12_add,
            par[0].block12_par_expand, par[1].block12_par_expand,
            par[0].block12_par_expand_BN, par[1].block12_par_expand_BN,
            par[0].block12_par_depth, par[1].block12_par_depth,
            par[0].block12_par_depth_BN, par[1].block12_par_depth_BN,
            par[0].block12_par_project, par[1].block12_par_project,
            par[0].block12_par_project_BN, par[1].block12_par_project_BN);

    // Block 11
    backprop_S1(14,         // input and output sizes
            64,             // input depth
            18, 27, 9,      // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 64,        // Expand and project factors
            block10_add,
            block11_expand, block11_expand_relu,
            block11_depth, block11_depth_relu,
            block11_project, block11_add,
            par[0].block11_par_expand, par[1].block11_par_expand,
            par[0].block11_par_expand_BN, par[1].block11_par_expand_BN,
            par[0].block11_par_depth, par[1].block11_par_depth,
            par[0].block11_par_depth_BN, par[1].block11_par_depth_BN,
            par[0].block11_par_project, par[1].block11_par_project,
            par[0].block11_par_project_BN, par[1].block11_par_project_BN);

    // Block 10
    backprop_S1(14,         // input and output sizes
            64,             // input depth
            16, 24, 8,      // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 64,        // Expand and project factors
            block9_add,
            block10_expand, block10_expand_relu,
            block10_depth, block10_depth_relu,
            block10_project, block10_add,
            par[0].block10_par_expand, par[1].block10_par_expand,
            par[0].block10_par_expand_BN, par[1].block10_par_expand_BN,
            par[0].block10_par_depth, par[1].block10_par_depth,
            par[0].block10_par_depth_BN, par[1].block10_par_depth_BN,
            par[0].block10_par_project, par[1].block10_par_project,
            par[0].block10_par_project_BN, par[1].block10_par_project_BN);

    // Block 9
    backprop_S1(14,         // input and output sizes
            64,             // input depth
            14, 21, 7,      // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 64,        // Expand and project factors
            block8_project_BN,
            block9_expand, block9_expand_relu,
            block9_depth, block9_depth_relu,
            block9_project, block9_add,
            par[0].block9_par_expand, par[1].block9_par_expand,
            par[0].block9_par_expand_BN, par[1].block9_par_expand_BN,
            par[0].block9_par_depth, par[1].block9_par_depth,
            par[0].block9_par_depth_BN, par[1].block9_par_depth_BN,
            par[0].block9_par_project, par[1].block9_par_project,
            par[0].block9_par_project_BN, par[1].block9_par_project_BN);

    // Block 8
    backprop_S2(28, 14,     // input and output sizes / depthwise factor
            32, 64,         // input and output depth / project factor
            12, 18, 6,      // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            192,            // Expand factor
            block7_add,
            block8_expand, block8_expand_relu,
            block8_depth, block8_depth_relu,
            block8_project, block8_project_BN,
            par[0].block8_par_expand, par[1].block8_par_expand,
            par[0].block8_par_expand_BN, par[1].block8_par_expand_BN,
            par[0].block8_par_depth, par[1].block8_par_depth,
            par[0].block8_par_depth_BN, par[1].block8_par_depth_BN,
            par[0].block8_par_project, par[1].block8_par_project,
            par[0].block8_par_project_BN, par[1].block8_par_project_BN);

    // Block 7
    backprop_S1(28,         // input and output sizes
            32,             // input depth
            10, 15, 5,      // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            192, 32,        // Expand and project factors
            block6_add,
            block7_expand, block7_expand_relu,
            block7_depth, block7_depth_relu,
            block7_project, block7_add,
            par[0].block7_par_expand, par[1].block7_par_expand,
            par[0].block7_par_expand_BN, par[1].block7_par_expand_BN,
            par[0].block7_par_depth, par[1].block7_par_depth,
            par[0].block7_par_depth_BN, par[1].block7_par_depth_BN,
            par[0].block7_par_project, par[1].block7_par_project,
            par[0].block7_par_project_BN, par[1].block7_par_project_BN);

    // Block 6
    backprop_S1(28,         // input and output sizes
            32,             // input depth
            8, 12, 4,      // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            192, 32,        // Expand and project factors
            block5_project_BN,
            block6_expand, block6_expand_relu,
            block6_depth, block6_depth_relu,
            block6_project, block6_add,
            par[0].block6_par_expand, par[1].block6_par_expand,
            par[0].block6_par_expand_BN, par[1].block6_par_expand_BN,
            par[0].block6_par_depth, par[1].block6_par_depth,
            par[0].block6_par_depth_BN, par[1].block6_par_depth_BN,
            par[0].block6_par_project, par[1].block6_par_project,
            par[0].block6_par_project_BN, par[1].block6_par_project_BN);

    // Block 5
    backprop_S2(56, 28,     // input and output sizes / depthwise factor
            24, 32,         // input and output depth / project factor
            6, 9, 3,        // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            144,            // Expand factor
            block4_add,
            block5_expand, block5_expand_relu,
            block5_depth, block5_depth_relu,
            block5_project, block5_project_BN,
            par[0].block5_par_expand, par[1].block5_par_expand,
            par[0].block5_par_expand_BN, par[1].block5_par_expand_BN,
            par[0].block5_par_depth, par[1].block5_par_depth,
            par[0].block5_par_depth_BN, par[1].block5_par_depth_BN,
            par[0].block5_par_project, par[1].block5_par_project,
            par[0].block5_par_project_BN, par[1].block5_par_project_BN);

    // Block 4
    backprop_S1(56,         // input and output sizes
            24,             // input depth
            4, 6, 2,        // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            144, 24,        // Expand and project factors
            block3_project_BN,
            block4_expand, block4_expand_relu,
            block4_depth, block4_depth_relu,
            block4_project, block4_add,
            par[0].block4_par_expand, par[1].block4_par_expand,
            par[0].block4_par_expand_BN, par[1].block4_par_expand_BN,
            par[0].block4_par_depth, par[1].block4_par_depth,
            par[0].block4_par_depth_BN, par[1].block4_par_depth_BN,
            par[0].block4_par_project, par[1].block4_par_project,
            par[0].block4_par_project_BN, par[1].block4_par_project_BN);

    // Block 3
    backprop_S2(112, 56,    // input and output sizes / depthwise factor
            16, 24,         // input and output depth / project factor
            2, 3, 1,        // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            96,             // Expand factor
            expanded_project_BN,
            block3_expand, block3_expand_relu,
            block3_depth, block3_depth_relu,
            block3_project, block3_project_BN,
            par[0].block3_par_expand, par[1].block3_par_expand,
            par[0].block3_par_expand_BN, par[1].block3_par_expand_BN,
            par[0].block3_par_depth, par[1].block3_par_depth,
            par[0].block3_par_depth_BN, par[1].block3_par_depth_BN,
            par[0].block3_par_project, par[1].block3_par_project,
            par[0].block3_par_project_BN, par[1].block3_par_project_BN);

    // Expanded Block
    backprop_bn(112, 16,
            expanded_project, expanded_project_BN,
            par[0].expanded_par_project_BN, par[1].expanded_par_project_BN,
            3);

    backprop_conv2d(112, 112, 1, 32, 16,
            expanded_relu, expanded_project,
            par[0].expanded_par_project, par[1].expanded_par_project,
            1, 0, 2);

    backprop_relu6(112, 32, // Dimensions
            expanded_relu   // Input and Output (error)
            );

    backprop_bn(112, 32,
            expanded_depth, expanded_relu,
            par[0].expanded_par_depth_BN, par[1].expanded_par_depth_BN,
            2);

    backprop_dw(112, 112, 3, 32,
            initial_relu, expanded_depth,
            par[0].expanded_par_depth, par[1].expanded_par_depth,
            1, 1, 1);

    // Initial Block
    backprop_relu6(112, 32, // Dimensions
            initial_relu    // Input and Output (error)
            );

    backprop_bn(112, 32,
            initial_conv2d, initial_relu,
            par[0].initial_par_BN, par[1].initial_par_BN,
            1);

    backprop_conv2d(224, 112, 3, 3, 32,
            image, initial_conv2d,
            par[0].initial_par_conv2d, par[1].initial_par_conv2d,
            2, 0, 1);
}
