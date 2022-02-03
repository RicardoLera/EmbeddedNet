#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "blocks.h"
#include "layers.h"
#include "operations.h"
#include "var.h"

void inference() {

    // Initial Block
    printf("Entering Initial Block: 224x224x3\n");

    conv2d(224, 112, 3, // input, output, kernel size
            2,          // stride
            0,          // pad input by one pixels de facto centering the kernel
            3, 32,      // number of input and output channels
            1,          // weight import index
            image, var.initial_conv2d, par[0].initial_par_conv2d);

    batch_normalization(112,    // input, output
            32,                 // number of channels
            1,                  // weight import index
            var.initial_conv2d, var.initial_relu, par[0].initial_par_BN);

    relu6(112,  // input, output
            32, // number of channels
            var.initial_relu);  // This one goes straight to "operations" because there is no parameter allocation necessary


    // Expanded Block
    printf("Entering Expanded Block: 112x112x32\n");

    S1depthwise(112, 3, // size and kernel size
            1,          // padding
            32,         // depth
            1,          // weight import index
            var.initial_relu, var.expanded_depth, par[0].expanded_par_depth);

    batch_normalization(112,    // input, output
            32,                 // number of channels
            2,                  // weight import index
            var.expanded_depth, var.expanded_relu, par[0].expanded_par_depth_BN);

    relu6(112,  // input, output
            32, // number of channels
            var.expanded_relu);

    conv2d(112, 112, 1, // input, output, kernel size
            1,          // stride
            0,          // pad input by one pixels de facto centering the kernel
            32, 16,     // number of input and output channels
            2,          // weight import index
            var.expanded_relu, var.expanded_project, par[0].expanded_par_project);

    batch_normalization(112,    // input, output
            16,                 // number of channels
            3,                  // weight import index
            var.expanded_project, var.expanded_project_BN, par[0].expanded_par_project_BN);


    // Block 3 - Stride 2 (Block 1 in Python)
    printf("Entering Block 3: 112x112x16\n");

    S2block(112, 56,    // input and output sizes / depthwise factor
            16, 24,     // input and output depth / project factor
            2, 3, 1,    // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            96,         // Expand factor
            var.expanded_project_BN,
            var.block3_expand, var.block3_expand_relu,
            var.block3_depth, var.block3_depth_relu,
            var.block3_project, var.block3_project_BN,
            par[0].block3_par_expand, par[0].block3_par_expand_BN,
            par[0].block3_par_depth, par[0].block3_par_depth_BN,
            par[0].block3_par_project, par[0].block3_par_project_BN);


    // Block 4 - Stride 1 (Block 2 in Python)
    printf("Entering Block 4: 56x56x24\n");

    S1block(56,         // input and output sizes
            24,         // input depth
            4, 6, 2,    // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            144, 24, 1, // Expand, project and add factors
            var.block3_project_BN,
            var.block4_expand, var.block4_expand_relu,
            var.block4_depth, var.block4_depth_relu,
            var.block4_project, var.block4_add,
            par[0].block4_par_expand, par[0].block4_par_expand_BN,
            par[0].block4_par_depth, par[0].block4_par_depth_BN,
            par[0].block4_par_project, par[0].block4_par_project_BN);


    // Block 5 - Stride 2 (Block 3 in Python)
    printf("Entering Block 5: 56x56x24\n");

    S2block(56, 28,     // input and output sizes / depthwise factor
            24, 32,     // input and output depth / project factor
            6, 9, 3,    // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            144,        // Expand factor
            var.block4_add,
            var.block5_expand, var.block5_expand_relu,
            var.block5_depth, var.block5_depth_relu,
            var.block5_project, var.block5_project_BN,
            par[0].block5_par_expand, par[0].block5_par_expand_BN,
            par[0].block5_par_depth, par[0].block5_par_depth_BN,
            par[0].block5_par_project, par[0].block5_par_project_BN);


    // Block 6 - Stride 1 (Block 4 in Python)
    printf("Entering Block 6: 28x28x32\n");

    S1block(28,         // input and output sizes
            32,         // input depth
            8, 12, 4,   // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            192, 32, 1, // Expand, project and add factors
            var.block5_project_BN,
            var.block6_expand, var.block6_expand_relu,
            var.block6_depth, var.block6_depth_relu,
            var.block6_project, var.block6_add,
            par[0].block6_par_expand, par[0].block6_par_expand_BN,
            par[0].block6_par_depth, par[0].block6_par_depth_BN,
            par[0].block6_par_project, par[0].block6_par_project_BN);


    // Block 7 - Stride 1 (Block 5 in Python)
    printf("Entering Block 7: 28x28x32\n");

    S1block(28,         // input and output sizes
            32,         // input depth
            10, 15, 5,  // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            192, 32, 1, // Expand, project and add factors
            var.block6_add,
            var.block7_expand, var.block7_expand_relu,
            var.block7_depth, var.block7_depth_relu,
            var.block7_project, var.block7_add,
            par[0].block7_par_expand, par[0].block7_par_expand_BN,
            par[0].block7_par_depth, par[0].block7_par_depth_BN,
            par[0].block7_par_project, par[0].block7_par_project_BN);


    // Block 8 - Stride 2 (Block 6 in Python)
    printf("Entering Block 8: 28x28x32\n");

    S2block(28, 14,     // input and output sizes / depthwise factor
            32, 64,     // input and output depth / project factor
            12, 18, 6,  // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            192,        // Expand factor
            var.block7_add,
            var.block8_expand, var.block8_expand_relu,
            var.block8_depth, var.block8_depth_relu,
            var.block8_project, var.block8_project_BN,
            par[0].block8_par_expand, par[0].block8_par_expand_BN,
            par[0].block8_par_depth, par[0].block8_par_depth_BN,
            par[0].block8_par_project, par[0].block8_par_project_BN);


    // Block 9 - Stride 1 (Block 7 in Python)
    printf("Entering Block 9: 14x14x64\n");

    S1block(14,         // input and output sizes
            64,         // input depth
            14, 21, 7,  // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 64, 1, // Expand, project and add factors
            var.block8_project_BN,
            var.block9_expand, var.block9_expand_relu,
            var.block9_depth, var.block9_depth_relu,
            var.block9_project, var.block9_add,
            par[0].block9_par_expand, par[0].block9_par_expand_BN,
            par[0].block9_par_depth, par[0].block9_par_depth_BN,
            par[0].block9_par_project, par[0].block9_par_project_BN);


    // Block 10 - Stride 1 (Block 8 in Python)
    printf("Entering Block 10: 14x14x64\n");

    S1block(14,         // input and output sizes
            64,         // input depth
            16, 24, 8,  // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 64, 1, // Expand, project and add factors
            var.block9_add,
            var.block10_expand, var.block10_expand_relu,
            var.block10_depth, var.block10_depth_relu,
            var.block10_project, var.block10_add,
            par[0].block10_par_expand, par[0].block10_par_expand_BN,
            par[0].block10_par_depth, par[0].block10_par_depth_BN,
            par[0].block10_par_project, par[0].block10_par_project_BN);


    // Block 11 - Stride 1 (Block 9 in Python)
    printf("Entering Block 11: 14x14x64\n");

    S1block(14,         // input and output sizes
            64,         // input depth
            18, 27, 9,  // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 64, 1, // Expand, project and add factors
            var.block10_add,
            var.block11_expand, var.block11_expand_relu,
            var.block11_depth, var.block11_depth_relu,
            var.block11_project, var.block11_add,
            par[0].block11_par_expand, par[0].block11_par_expand_BN,
            par[0].block11_par_depth, par[0].block11_par_depth_BN,
            par[0].block11_par_project, par[0].block11_par_project_BN);


    // Block 12 - Stride 1 - Over-Project (Block 10 in Python)
    printf("Entering Block 12: 14x14x64\n");

    S1block(14,         // input and output sizes
            64,         // input depth
            20, 30, 10, // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 96, 0, // Expand, project and add factors
            var.block11_add,
            var.block12_expand, var.block12_expand_relu,
            var.block12_depth, var.block12_depth_relu,
            var.block12_project, var.block12_add,
            par[0].block12_par_expand, par[0].block12_par_expand_BN,
            par[0].block12_par_depth, par[0].block12_par_depth_BN,
            par[0].block12_par_project, par[0].block12_par_project_BN);


    // Block 13 - Stride 1 (Block 11 in Python)
    printf("Entering Block 13: 14x14x96\n");

    S1block(14,         // input and output sizes
            96,         // input depth
            22, 33, 11, // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            576, 96, 1, // Expand, project and add factors
            var.block12_add,
            var.block13_expand, var.block13_expand_relu,
            var.block13_depth, var.block13_depth_relu,
            var.block13_project, var.block13_add,
            par[0].block13_par_expand, par[0].block13_par_expand_BN,
            par[0].block13_par_depth, par[0].block13_par_depth_BN,
            par[0].block13_par_project, par[0].block13_par_project_BN);


    // Block 14 - Stride 1 (Block 12 in Python)
    printf("Entering Block 14: 14x14x96\n");

    S1block(14,         // input and output sizes
            96,         // input depth
            24, 36, 12, // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            576, 96, 1, // Expand, project and add factors
            var.block13_add,
            var.block14_expand, var.block14_expand_relu,
            var.block14_depth, var.block14_depth_relu,
            var.block14_project, var.block14_add,
            par[0].block14_par_expand, par[0].block14_par_expand_BN,
            par[0].block14_par_depth, par[0].block14_par_depth_BN,
            par[0].block14_par_project, par[0].block14_par_project_BN);


    // Block 15 - Stride 2 (Block 13 in Python)
    printf("Entering Block 15: 14x14x96\n");

    S2block(14, 7,      // input and output sizes / depthwise factor
            96, 160,    // input depth / project factor
            26, 39, 13, // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            576,        // Expand factor
            var.block14_add,
            var.block15_expand, var.block15_expand_relu,
            var.block15_depth, var.block15_depth_relu,
            var.block15_project, var.block15_project_BN,
            par[0].block15_par_expand, par[0].block15_par_expand_BN,
            par[0].block15_par_depth, par[0].block15_par_depth_BN,
            par[0].block15_par_project, par[0].block15_par_project_BN);


    // Block 16 - Stride 1 (Block 14 in Python)
    printf("Entering Block 16: 7x7x160\n");

    S1block(7,              // input and output sizes
            160,            // input depth
            28, 42, 14,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            960, 160, 1,    // Expand, project and add factors
            var.block15_project_BN,
            var.block16_expand, var.block16_expand_relu,
            var.block16_depth, var.block16_depth_relu,
            var.block16_project, var.block16_add,
            par[0].block16_par_expand, par[0].block16_par_expand_BN,
            par[0].block16_par_depth, par[0].block16_par_depth_BN,
            par[0].block16_par_project, par[0].block16_par_project_BN);


    // Block 17 - Stride 1 (Block 15 in Python)
    printf("Entering Block 17: 7x7x160\n");

    S1block(7,              // input and output sizes
            160,            // input depth
            30, 45, 15,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            960, 160, 1,    // Expand, project and add factors
            var.block16_add,
            var.block17_expand, var.block17_expand_relu,
            var.block17_depth, var.block17_depth_relu,
            var.block17_project, var.block17_add,
            par[0].block17_par_expand, par[0].block17_par_expand_BN,
            par[0].block17_par_depth, par[0].block17_par_depth_BN,
            par[0].block17_par_project, par[0].block17_par_project_BN);


    // Block 18 - Stride 1 - Over-Project (Block 16 in Python)
    printf("Entering Block 18: 7x7x160\n");

    S1block(7,              // input and output sizes
            160,            // input depth
            32, 48, 16,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            960, 320, 0,    // Expand, project and add factors
            var.block17_add,
            var.block18_expand, var.block18_expand_relu,
            var.block18_depth, var.block18_depth_relu,
            var.block18_project, var.block18_add,
            par[0].block18_par_expand, par[0].block18_par_expand_BN,
            par[0].block18_par_depth, par[0].block18_par_depth_BN,
            par[0].block18_par_project, par[0].block18_par_project_BN);


    // Final Block
    printf("Entering Final Block: 7x7x320\n");

    conv2d(7, 7, 1,     // input, output, kernel size
            1,          // stride
            0,          // pad input by one pixels de facto centering the kernel
            320, 1280,  // number of input and output channels
            35,         // weight import index
            var.block18_add, var.final_conv2d, par[0].final_par_conv2d);

    batch_normalization(7,  // input, output
            1280,           // number of channels
            52,             // weight import index
            var.final_conv2d, var.final_conv2d_relu, par[0].final_par_conv2d_BN);

    relu6(7,        // input, output
            1280,   // number of channels
            var.final_conv2d_relu);

    avgpool(var.final_conv2d_relu, var.final_pooling);

    fully_connected(var.final_pooling, var.predictions, par[0].final_par_fc_w, par[0].final_par_fc_b);

    softmax(var.predictions, 1000);

    decode(var.predictions);

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
    backprop_fc(var.final_pooling,  // Input of layer
            var.predictions,        // Output of layer (recycled as backpropagated error)
            label,                  // Correct prediction
            par[0].final_par_fc_w, par[0].final_par_fc_b,
            par[1].final_par_fc_w, par[1].final_par_fc_b);

    if (strcmp(frz,"fc") == 0) {exit(0);}

    backprop_avrgpool(var.final_conv2d_relu,    // Input of layer
            var.final_pooling);                 // Output of layer

    backprop_relu6(7, 1280,         // Size and depth
            var.final_conv2d_relu); // Input and Output


    backprop_bn(7, 1280,    // Size and Depth
            var.final_conv2d, var.final_conv2d_relu,
            par[0].final_par_conv2d_BN, par[1].final_par_conv2d_BN,
            52);            // Import index

    backprop_conv2d(7, 7, 1, 320, 1280, // Line size, column size, kernel size, input depth, output depth
            var.block18_add, var.final_conv2d,
            par[0].final_par_conv2d, par[1].final_par_conv2d,
            1, 0, 35);                  // Stride, padding and import index

    if (strcmp(frz,"b18") == 0) {exit(0);}

    // Block 18
    backprop_S1(7,          // input and output sizes
            160,            // input depth
            32, 48, 16,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            960, 320,       // Expand and project factors
            var.block17_add,
            var.block18_expand, var.block18_expand_relu,
            var.block18_depth, var.block18_depth_relu,
            var.block18_project, var.block18_add,
            par[0].block18_par_expand, par[1].block18_par_expand,
            par[0].block18_par_expand_BN, par[1].block18_par_expand_BN,
            par[0].block18_par_depth, par[1].block18_par_depth,
            par[0].block18_par_depth_BN, par[1].block18_par_depth_BN,
            par[0].block18_par_project, par[1].block18_par_project,
            par[0].block18_par_project_BN, par[1].block18_par_project_BN);

    if (strcmp(frz,"b17") == 0) {exit(0);}

    // Block 17
    backprop_S1(7,          // input and output sizes
            160,            // input depth
            30, 45, 15,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            960, 160,       // Expand and project factors
            var.block16_add,
            var.block17_expand, var.block17_expand_relu,
            var.block17_depth, var.block17_depth_relu,
            var.block17_project, var.block17_add,
            par[0].block17_par_expand, par[1].block17_par_expand,
            par[0].block17_par_expand_BN, par[1].block17_par_expand_BN,
            par[0].block17_par_depth, par[1].block17_par_depth,
            par[0].block17_par_depth_BN, par[1].block17_par_depth_BN,
            par[0].block17_par_project, par[1].block17_par_project,
            par[0].block17_par_project_BN, par[1].block17_par_project_BN);

    if (strcmp(frz,"b16") == 0) {exit(0);}

    // Block 16
    backprop_S1(7,          // input and output sizes
            160,            // input depth
            28, 42, 14,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            960, 160,       // Expand and project factors
            var.block15_project_BN,
            var.block16_expand, var.block16_expand_relu,
            var.block16_depth, var.block16_depth_relu,
            var.block16_project, var.block16_add,
            par[0].block16_par_expand, par[1].block16_par_expand,
            par[0].block16_par_expand_BN, par[1].block16_par_expand_BN,
            par[0].block16_par_depth, par[1].block16_par_depth,
            par[0].block16_par_depth_BN, par[1].block16_par_depth_BN,
            par[0].block16_par_project, par[1].block16_par_project,
            par[0].block16_par_project_BN, par[1].block16_par_project_BN);

    if (strcmp(frz,"b15") == 0) {exit(0);}

    // Block 15
    backprop_S2(14, 7,      // input and output sizes / depthwise factor
            96, 160,        // input and output depth / project factor
            26, 39, 13,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            576,            // Expand factor
            var.block14_add,
            var.block15_expand, var.block15_expand_relu,
            var.block15_depth, var.block15_depth_relu,
            var.block15_project, var.block15_project_BN,
            par[0].block15_par_expand, par[1].block15_par_expand,
            par[0].block15_par_expand_BN, par[1].block15_par_expand_BN,
            par[0].block15_par_depth, par[1].block15_par_depth,
            par[0].block15_par_depth_BN, par[1].block15_par_depth_BN,
            par[0].block15_par_project, par[1].block15_par_project,
            par[0].block15_par_project_BN, par[1].block15_par_project_BN);

    if (strcmp(frz,"b14") == 0) {exit(0);}

    // Block 14
    backprop_S1(14,         // input and output sizes
            96,             // input depth
            24, 36, 12,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            576, 96,        // Expand and project factors
            var.block13_add,
            var.block14_expand, var.block14_expand_relu,
            var.block14_depth, var.block14_depth_relu,
            var.block14_project, var.block14_add,
            par[0].block14_par_expand, par[1].block14_par_expand,
            par[0].block14_par_expand_BN, par[1].block14_par_expand_BN,
            par[0].block14_par_depth, par[1].block14_par_depth,
            par[0].block14_par_depth_BN, par[1].block14_par_depth_BN,
            par[0].block14_par_project, par[1].block14_par_project,
            par[0].block14_par_project_BN, par[1].block14_par_project_BN);

    if (strcmp(frz,"b13") == 0) {exit(0);}

    // Block 13
    backprop_S1(14,         // input and output sizes
            96,             // input depth
            22, 33, 11,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            576, 96,        // Expand and project factors
            var.block12_add,
            var.block13_expand, var.block13_expand_relu,
            var.block13_depth, var.block13_depth_relu,
            var.block13_project, var.block13_add,
            par[0].block13_par_expand, par[1].block13_par_expand,
            par[0].block13_par_expand_BN, par[1].block13_par_expand_BN,
            par[0].block13_par_depth, par[1].block13_par_depth,
            par[0].block13_par_depth_BN, par[1].block13_par_depth_BN,
            par[0].block13_par_project, par[1].block13_par_project,
            par[0].block13_par_project_BN, par[1].block13_par_project_BN);

    if (strcmp(frz,"b12") == 0) {exit(0);}

    // Block 12
    backprop_S1(14,         // input and output sizes
            64,             // input depth
            20, 30, 10,     // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 96,        // Expand and project factors
            var.block11_add,
            var.block12_expand, var.block12_expand_relu,
            var.block12_depth, var.block12_depth_relu,
            var.block12_project, var.block12_add,
            par[0].block12_par_expand, par[1].block12_par_expand,
            par[0].block12_par_expand_BN, par[1].block12_par_expand_BN,
            par[0].block12_par_depth, par[1].block12_par_depth,
            par[0].block12_par_depth_BN, par[1].block12_par_depth_BN,
            par[0].block12_par_project, par[1].block12_par_project,
            par[0].block12_par_project_BN, par[1].block12_par_project_BN);

    if (strcmp(frz,"b11") == 0) {exit(0);}

    // Block 11
    backprop_S1(14,         // input and output sizes
            64,             // input depth
            18, 27, 9,      // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 64,        // Expand and project factors
            var.block10_add,
            var.block11_expand, var.block11_expand_relu,
            var.block11_depth, var.block11_depth_relu,
            var.block11_project, var.block11_add,
            par[0].block11_par_expand, par[1].block11_par_expand,
            par[0].block11_par_expand_BN, par[1].block11_par_expand_BN,
            par[0].block11_par_depth, par[1].block11_par_depth,
            par[0].block11_par_depth_BN, par[1].block11_par_depth_BN,
            par[0].block11_par_project, par[1].block11_par_project,
            par[0].block11_par_project_BN, par[1].block11_par_project_BN);

    if (strcmp(frz,"b10") == 0) {exit(0);}

    // Block 10
    backprop_S1(14,         // input and output sizes
            64,             // input depth
            16, 24, 8,      // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 64,        // Expand and project factors
            var.block9_add,
            var.block10_expand, var.block10_expand_relu,
            var.block10_depth, var.block10_depth_relu,
            var.block10_project, var.block10_add,
            par[0].block10_par_expand, par[1].block10_par_expand,
            par[0].block10_par_expand_BN, par[1].block10_par_expand_BN,
            par[0].block10_par_depth, par[1].block10_par_depth,
            par[0].block10_par_depth_BN, par[1].block10_par_depth_BN,
            par[0].block10_par_project, par[1].block10_par_project,
            par[0].block10_par_project_BN, par[1].block10_par_project_BN);

    if (strcmp(frz,"b9") == 0) {exit(0);}

    // Block 9
    backprop_S1(14,         // input and output sizes
            64,             // input depth
            14, 21, 7,      // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            384, 64,        // Expand and project factors
            var.block8_project_BN,
            var.block9_expand, var.block9_expand_relu,
            var.block9_depth, var.block9_depth_relu,
            var.block9_project, var.block9_add,
            par[0].block9_par_expand, par[1].block9_par_expand,
            par[0].block9_par_expand_BN, par[1].block9_par_expand_BN,
            par[0].block9_par_depth, par[1].block9_par_depth,
            par[0].block9_par_depth_BN, par[1].block9_par_depth_BN,
            par[0].block9_par_project, par[1].block9_par_project,
            par[0].block9_par_project_BN, par[1].block9_par_project_BN);

    if (strcmp(frz,"b8") == 0) {exit(0);}

    // Block 8
    backprop_S2(28, 14,     // input and output sizes / depthwise factor
            32, 64,         // input and output depth / project factor
            12, 18, 6,      // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            192,            // Expand factor
            var.block7_add,
            var.block8_expand, var.block8_expand_relu,
            var.block8_depth, var.block8_depth_relu,
            var.block8_project, var.block8_project_BN,
            par[0].block8_par_expand, par[1].block8_par_expand,
            par[0].block8_par_expand_BN, par[1].block8_par_expand_BN,
            par[0].block8_par_depth, par[1].block8_par_depth,
            par[0].block8_par_depth_BN, par[1].block8_par_depth_BN,
            par[0].block8_par_project, par[1].block8_par_project,
            par[0].block8_par_project_BN, par[1].block8_par_project_BN);

    if (strcmp(frz,"b7") == 0) {exit(0);}

    // Block 7
    backprop_S1(28,         // input and output sizes
            32,             // input depth
            10, 15, 5,      // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            192, 32,        // Expand and project factors
            var.block6_add,
            var.block7_expand, var.block7_expand_relu,
            var.block7_depth, var.block7_depth_relu,
            var.block7_project, var.block7_add,
            par[0].block7_par_expand, par[1].block7_par_expand,
            par[0].block7_par_expand_BN, par[1].block7_par_expand_BN,
            par[0].block7_par_depth, par[1].block7_par_depth,
            par[0].block7_par_depth_BN, par[1].block7_par_depth_BN,
            par[0].block7_par_project, par[1].block7_par_project,
            par[0].block7_par_project_BN, par[1].block7_par_project_BN);

    if (strcmp(frz,"b6") == 0) {exit(0);}

    // Block 6
    backprop_S1(28,         // input and output sizes
            32,             // input depth
            8, 12, 4,      // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            192, 32,        // Expand and project factors
            var.block5_project_BN,
            var.block6_expand, var.block6_expand_relu,
            var.block6_depth, var.block6_depth_relu,
            var.block6_project, var.block6_add,
            par[0].block6_par_expand, par[1].block6_par_expand,
            par[0].block6_par_expand_BN, par[1].block6_par_expand_BN,
            par[0].block6_par_depth, par[1].block6_par_depth,
            par[0].block6_par_depth_BN, par[1].block6_par_depth_BN,
            par[0].block6_par_project, par[1].block6_par_project,
            par[0].block6_par_project_BN, par[1].block6_par_project_BN);

    if (strcmp(frz,"b5") == 0) {exit(0);}

    // Block 5
    backprop_S2(56, 28,     // input and output sizes / depthwise factor
            24, 32,         // input and output depth / project factor
            6, 9, 3,        // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            144,            // Expand factor
            var.block4_add,
            var.block5_expand, var.block5_expand_relu,
            var.block5_depth, var.block5_depth_relu,
            var.block5_project, var.block5_project_BN,
            par[0].block5_par_expand, par[1].block5_par_expand,
            par[0].block5_par_expand_BN, par[1].block5_par_expand_BN,
            par[0].block5_par_depth, par[1].block5_par_depth,
            par[0].block5_par_depth_BN, par[1].block5_par_depth_BN,
            par[0].block5_par_project, par[1].block5_par_project,
            par[0].block5_par_project_BN, par[1].block5_par_project_BN);

    if (strcmp(frz,"b4") == 0) {exit(0);}

    // Block 4
    backprop_S1(56,         // input and output sizes
            24,             // input depth
            4, 6, 2,        // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            144, 24,        // Expand and project factors
            var.block3_project_BN,
            var.block4_expand, var.block4_expand_relu,
            var.block4_depth, var.block4_depth_relu,
            var.block4_project, var.block4_add,
            par[0].block4_par_expand, par[1].block4_par_expand,
            par[0].block4_par_expand_BN, par[1].block4_par_expand_BN,
            par[0].block4_par_depth, par[1].block4_par_depth,
            par[0].block4_par_depth_BN, par[1].block4_par_depth_BN,
            par[0].block4_par_project, par[1].block4_par_project,
            par[0].block4_par_project_BN, par[1].block4_par_project_BN);

    if (strcmp(frz,"b3") == 0) {exit(0);}

    // Block 3
    backprop_S2(112, 56,    // input and output sizes / depthwise factor
            16, 24,         // input and output depth / project factor
            2, 3, 1,        // 2D convolution, Batch Normalization and Depthwise CURRENT weight import indexes
            96,             // Expand factor
            var.expanded_project_BN,
            var.block3_expand, var.block3_expand_relu,
            var.block3_depth, var.block3_depth_relu,
            var.block3_project, var.block3_project_BN,
            par[0].block3_par_expand, par[1].block3_par_expand,
            par[0].block3_par_expand_BN, par[1].block3_par_expand_BN,
            par[0].block3_par_depth, par[1].block3_par_depth,
            par[0].block3_par_depth_BN, par[1].block3_par_depth_BN,
            par[0].block3_par_project, par[1].block3_par_project,
            par[0].block3_par_project_BN, par[1].block3_par_project_BN);

    if (strcmp(frz,"exp") == 0) {exit(0);}

    // Expanded Block
    backprop_bn(112, 16,    // Size and depth
            var.expanded_project, var.expanded_project_BN,
            par[0].expanded_par_project_BN, par[1].expanded_par_project_BN,
            3);             // Import index

    backprop_conv2d(112, 112, 1, 32, 16,    // Line size, column size, kernel size, input depth, output depth
            var.expanded_relu, var.expanded_project,
            par[0].expanded_par_project, par[1].expanded_par_project,
            1, 0, 2);                       // Stride, padding and import index

    backprop_relu6(112, 32,     // Size and depth
            var.expanded_relu);

    backprop_bn(112, 32,    // Size and depth
            var.expanded_depth, var.expanded_relu,
            par[0].expanded_par_depth_BN, par[1].expanded_par_depth_BN,
            2);             // Import index

    backprop_dw(112, 112, 3, 32,    // Input size, output size, kernel size, depth
            var.initial_relu, var.expanded_depth,
            par[0].expanded_par_depth, par[1].expanded_par_depth,
            1, 1, 1);               // Stride, padding and import index

    // Initial Block
    backprop_relu6(112, 32,     // Size and depth
            var.initial_relu);

    backprop_bn(112, 32,    // Size and depth
            var.initial_conv2d, var.initial_relu,
            par[0].initial_par_BN, par[1].initial_par_BN,
            1);             // Import index

    backprop_conv2d(224, 112, 3, 3, 32, // Line size, column size, kernel size, input depth, output depth
            image, var.initial_conv2d,
            par[0].initial_par_conv2d, par[1].initial_par_conv2d,
            2, 0, 1);                   // Stride, padding and import index
}
