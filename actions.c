#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "blocks.h"
#include "layers.h"
#include "operations.h"
#include "data_manip.h"
#include "var.h"

void inference(int c, float predictions[c], float fc_w[1280][c][2], float fc_b[c][2]) {

    printf("Inferring...\n");

    // Initial Block
    //printf("Entering Initial Block: 224x224x3\n");

    conv2d(224, 112, 3,                         // Input size, output size, kernel size
            2,                                  // Stride
            0,                                  // Padding
            3, 32,                              // Number of input and output channels
            1,                                  // Weight import index
            image,                              // Input
            var.initial_conv2d,                 // Output
            par[0].initial_par_conv2d);         // Weights

    batch_normalization(112,                    // Input/output size
            32,                                 // Number of channels
            1,                                  // Parameter import index
            var.initial_conv2d,                 // Input
            var.initial_relu,                   // Output
            par[0].initial_par_BN);             // Parameters

    relu6(112,                                  // Input/output size
            32,                                 // Number of channels
            var.initial_relu);                  // Input/output
    // This one goes straight to "operations" because there is no parameter allocation necessary


    // Expanded Block
    //printf("Entering Expanded Block: 112x112x32\n");

    S1depthwise(112, 3,                         // Size and kernel size
            1,                                  // Padding
            32,                                 // Depth
            1,                                  // Weight import index
            var.initial_relu,                   // Input
            var.expanded_depth,                 // Output
            par[0].expanded_par_depth);         // Weights

    batch_normalization(112,                    // Input/output size
            32,                                 // Number of channels
            2,                                  // Parameter import index
            var.expanded_depth,                 // Input
            var.expanded_relu,                  // Output
            par[0].expanded_par_depth_BN);      // Parameters

    relu6(112,                                  // Input/output size
            32,                                 // Number of channels
            var.expanded_relu);                 // Input/output

    conv2d(112, 112, 1,                         // Input size, output size, kernel size
            1,                                  // Stride
            0,                                  // Padding
            32, 16,                             // Number of input and output channels
            2,                                  // Weight import index
            var.expanded_relu,                  // Input
            var.expanded_project,               // Output
            par[0].expanded_par_project);       // Weights

    batch_normalization(112,                    // Input/output size
            16,                                 // Number of channels
            3,                                  // Parameter import index
            var.expanded_project,               // Input
            var.expanded_project_BN,            // Output
            par[0].expanded_par_project_BN);    // Parameters


    // Block 3 - Stride 2 (Block 1 in Python)
    //printf("Entering Block 3: 112x112x16\n");

    S2block(112, 56,    // Input and output size / Depthwise factor
            16, 24,     // Input and output depth / Project factor
            2, 3, 1,    // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            96,         // Expand factor
            var.expanded_project_BN,                                    // Input
            var.block3_expand,          var.block3_expand_relu,         // Expand output
            var.block3_depth,           var.block3_depth_relu,          // Depthwise output
            var.block3_project,         var.block3_project_BN,          // Project output
            par[0].block3_par_expand,   par[0].block3_par_expand_BN,    // Expand parameters
            par[0].block3_par_depth,    par[0].block3_par_depth_BN,     // Depthwise parameters
            par[0].block3_par_project,  par[0].block3_par_project_BN);  // Project parameters


    // Block 4 - Stride 1 (Block 2 in Python)
    //printf("Entering Block 4: 56x56x24\n");

    S1block(56,         // Input and output size
            24,         // Input depth
            4, 6, 2,    // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            144, 24, 1, // Expand, project and add factors
            var.block3_project_BN,                                      // Input
            var.block4_expand,          var.block4_expand_relu,         // Expand output
            var.block4_depth,           var.block4_depth_relu,          // Depthwise output
            var.block4_project,         var.block4_add,                 // Project output
            par[0].block4_par_expand,   par[0].block4_par_expand_BN,    // Expand parameters
            par[0].block4_par_depth,    par[0].block4_par_depth_BN,     // Depthwise parameters
            par[0].block4_par_project,  par[0].block4_par_project_BN);  // Project parameters


    // Block 5 - Stride 2 (Block 3 in Python)
    //printf("Entering Block 5: 56x56x24\n");

    S2block(56, 28,     // Input and output size / Depthwise factor
            24, 32,     // Input and output depth / Project factor
            6, 9, 3,    // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            144,        // Expand factor
            var.block4_add,                                             // Input
            var.block5_expand,          var.block5_expand_relu,         // Expand output
            var.block5_depth,           var.block5_depth_relu,          // Depthwise output
            var.block5_project,         var.block5_project_BN,          // Project output
            par[0].block5_par_expand,   par[0].block5_par_expand_BN,    // Expand parameters
            par[0].block5_par_depth,    par[0].block5_par_depth_BN,     // Depthwise parameters
            par[0].block5_par_project,  par[0].block5_par_project_BN);  // Project parameters


    // Block 6 - Stride 1 (Block 4 in Python)
    //printf("Entering Block 6: 28x28x32\n");

    S1block(28,         // Input and output size
            32,         // Input depth
            8, 12, 4,   // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            192, 32, 1, // Expand, project and add factors
            var.block5_project_BN,                                      // Input
            var.block6_expand,          var.block6_expand_relu,         // Expand output
            var.block6_depth,           var.block6_depth_relu,          // Depthwise output
            var.block6_project,         var.block6_add,                 // Project output
            par[0].block6_par_expand,   par[0].block6_par_expand_BN,    // Expand parameters
            par[0].block6_par_depth,    par[0].block6_par_depth_BN,     // Depthwise parameters
            par[0].block6_par_project,  par[0].block6_par_project_BN);  // Project parameters


    // Block 7 - Stride 1 (Block 5 in Python)
    //printf("Entering Block 7: 28x28x32\n");

    S1block(28,         // Input and output size
            32,         // Input depth
            10, 15, 5,  // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            192, 32, 1, // Expand, project and add factors
            var.block6_add,                                             // Input
            var.block7_expand,          var.block7_expand_relu,         // Expand output
            var.block7_depth,           var.block7_depth_relu,          // Depthwise output
            var.block7_project,         var.block7_add,                 // Project output
            par[0].block7_par_expand,   par[0].block7_par_expand_BN,    // Expand parameters
            par[0].block7_par_depth,    par[0].block7_par_depth_BN,     // Depthwise parameters
            par[0].block7_par_project,  par[0].block7_par_project_BN);  // Project parameters


    // Block 8 - Stride 2 (Block 6 in Python)
    //printf("Entering Block 8: 28x28x32\n");

    S2block(28, 14,     // Input and output size / depthwise factor
            32, 64,     // input and output depth / project factor
            12, 18, 6,  // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            192,        // Expand factor
            var.block7_add,                                             // Input
            var.block8_expand,          var.block8_expand_relu,         // Expand output
            var.block8_depth,           var.block8_depth_relu,          // Depthwise output
            var.block8_project,         var.block8_project_BN,          // Project output
            par[0].block8_par_expand,   par[0].block8_par_expand_BN,    // Expand parameters
            par[0].block8_par_depth,    par[0].block8_par_depth_BN,     // Depthwise parameters
            par[0].block8_par_project,  par[0].block8_par_project_BN);  // Project parameters


    // Block 9 - Stride 1 (Block 7 in Python)
    //printf("Entering Block 9: 14x14x64\n");

    S1block(14,         // Input and output size
            64,         // Input depth
            14, 21, 7,  // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            384, 64, 1, // Expand, project and add factors
            var.block8_project_BN,                                      // Input
            var.block9_expand,          var.block9_expand_relu,         // Expand output
            var.block9_depth,           var.block9_depth_relu,          // Depthwise output
            var.block9_project,         var.block9_add,                 // Project output
            par[0].block9_par_expand,   par[0].block9_par_expand_BN,    // Expand parameters
            par[0].block9_par_depth,    par[0].block9_par_depth_BN,     // Depthwise parameters
            par[0].block9_par_project,  par[0].block9_par_project_BN);  // Project parameters


    // Block 10 - Stride 1 (Block 8 in Python)
    //printf("Entering Block 10: 14x14x64\n");

    S1block(14,         // Input and output size
            64,         // Input depth
            16, 24, 8,  // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            384, 64, 1, // Expand, project and add factors
            var.block9_add,                                             // Input
            var.block10_expand,         var.block10_expand_relu,        // Expand output
            var.block10_depth,          var.block10_depth_relu,         // Depthwise output
            var.block10_project,        var.block10_add,                // Project output
            par[0].block10_par_expand,  par[0].block10_par_expand_BN,   // Expand parameters
            par[0].block10_par_depth,   par[0].block10_par_depth_BN,    // Depthwise parameters
            par[0].block10_par_project, par[0].block10_par_project_BN); // Project parameters


    // Block 11 - Stride 1 (Block 9 in Python)
    //printf("Entering Block 11: 14x14x64\n");

    S1block(14,         // Input and output size
            64,         // Input depth
            18, 27, 9,  // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            384, 64, 1, // Expand, project and add factors
            var.block10_add,                                            // Input
            var.block11_expand,         var.block11_expand_relu,        // Expand output
            var.block11_depth,          var.block11_depth_relu,         // Depthwise output
            var.block11_project,        var.block11_add,                // Project output
            par[0].block11_par_expand,  par[0].block11_par_expand_BN,   // Expand parameters
            par[0].block11_par_depth,   par[0].block11_par_depth_BN,    // Depthwise parameters
            par[0].block11_par_project, par[0].block11_par_project_BN); // Project parameters


    // Block 12 - Stride 1 - Over-Project (Block 10 in Python)
    //printf("Entering Block 12: 14x14x64\n");

    S1block(14,         // Input and output size
            64,         // Input depth
            20, 30, 10, // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            384, 96, 0, // Expand, project and add factors
            var.block11_add,                                            // Input
            var.block12_expand,         var.block12_expand_relu,        // Expand output
            var.block12_depth,          var.block12_depth_relu,         // Depthwise output
            var.block12_project,        var.block12_add,                // Project output
            par[0].block12_par_expand,  par[0].block12_par_expand_BN,   // Expand parameters
            par[0].block12_par_depth,   par[0].block12_par_depth_BN,    // Depthwise parameters
            par[0].block12_par_project, par[0].block12_par_project_BN); // Project parameters


    // Block 13 - Stride 1 (Block 11 in Python)
    //printf("Entering Block 13: 14x14x96\n");

    S1block(14,         // Input and output size
            96,         // Input depth
            22, 33, 11, // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            576, 96, 1, // Expand, project and add factors
            var.block12_add,                                            // Input
            var.block13_expand,         var.block13_expand_relu,        // Expand output
            var.block13_depth,          var.block13_depth_relu,         // Depthwise output
            var.block13_project,        var.block13_add,                // Project output
            par[0].block13_par_expand,  par[0].block13_par_expand_BN,   // Expand parameters
            par[0].block13_par_depth,   par[0].block13_par_depth_BN,    // Depthwise parameters
            par[0].block13_par_project, par[0].block13_par_project_BN); // Project parameters


    // Block 14 - Stride 1 (Block 12 in Python)
    //printf("Entering Block 14: 14x14x96\n");

    S1block(14,         // Input and output size
            96,         // Input depth
            24, 36, 12, // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            576, 96, 1, // Expand, project and add factors
            var.block13_add,                                            // Input
            var.block14_expand,         var.block14_expand_relu,        // Expand output
            var.block14_depth,          var.block14_depth_relu,         // Depthwise output
            var.block14_project,        var.block14_add,                // Project output
            par[0].block14_par_expand,  par[0].block14_par_expand_BN,   // Expand parameters
            par[0].block14_par_depth,   par[0].block14_par_depth_BN,    // Depthwise parameters
            par[0].block14_par_project, par[0].block14_par_project_BN); // Project parameters


    // Block 15 - Stride 2 (Block 13 in Python)
    //printf("Entering Block 15: 14x14x96\n");

    S2block(14, 7,      // Input and output size / depthwise factor
            96, 160,    // Input depth / project factor
            26, 39, 13, // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            576,        // Expand factor
            var.block14_add,                                            // Input
            var.block15_expand,         var.block15_expand_relu,        // Expand output
            var.block15_depth,          var.block15_depth_relu,         // Depthwise output
            var.block15_project,        var.block15_project_BN,         // Project output
            par[0].block15_par_expand,  par[0].block15_par_expand_BN,   // Expand parameters
            par[0].block15_par_depth,   par[0].block15_par_depth_BN,    // Depthwise parameters
            par[0].block15_par_project, par[0].block15_par_project_BN); // Project parameters


    // Block 16 - Stride 1 (Block 14 in Python)
    //printf("Entering Block 16: 7x7x160\n");

    S1block(7,              // Input and output size
            160,            // Input depth
            28, 42, 14,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            960, 160, 1,    // Expand, project and add factors
            var.block15_project_BN,                                     // Input
            var.block16_expand,         var.block16_expand_relu,        // Expand output
            var.block16_depth,          var.block16_depth_relu,         // Depthwise output
            var.block16_project,        var.block16_add,                // Project output
            par[0].block16_par_expand,  par[0].block16_par_expand_BN,   // Expand parameters
            par[0].block16_par_depth,   par[0].block16_par_depth_BN,    // Depthwise parameters
            par[0].block16_par_project, par[0].block16_par_project_BN); // Project parameters


    // Block 17 - Stride 1 (Block 15 in Python)
    //printf("Entering Block 17: 7x7x160\n");

    S1block(7,              // Input and output size
            160,            // Input depth
            30, 45, 15,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            960, 160, 1,    // Expand, project and add factors
            var.block16_add,                                            // Input
            var.block17_expand,         var.block17_expand_relu,        // Expand output
            var.block17_depth,          var.block17_depth_relu,         // Depthwise output
            var.block17_project,        var.block17_add,                // Project output
            par[0].block17_par_expand,  par[0].block17_par_expand_BN,   // Expand parameters
            par[0].block17_par_depth,   par[0].block17_par_depth_BN,    // Depthwise parameters
            par[0].block17_par_project, par[0].block17_par_project_BN); // Project parameters


    // Block 18 - Stride 1 - Over-Project (Block 16 in Python)
    //printf("Entering Block 18: 7x7x160\n");

    S1block(7,              // Input and output size
            160,            // Input depth
            32, 48, 16,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            960, 320, 0,    // Expand, project and add factors
            var.block17_add,                                            // Input
            var.block18_expand,         var.block18_expand_relu,        // Expand output
            var.block18_depth,          var.block18_depth_relu,         // Depthwise output
            var.block18_project,        var.block18_add,                // Project output
            par[0].block18_par_expand,  par[0].block18_par_expand_BN,   // Expand parameters
            par[0].block18_par_depth,   par[0].block18_par_depth_BN,    // Depthwise parameters
            par[0].block18_par_project, par[0].block18_par_project_BN); // Project parameters


    // Final Block
    //printf("Entering Final Block: 7x7x320\n");

    conv2d(7, 7, 1,                         // Input size, output size, kernel size
            1,                              // Stride
            0,                              // Padding
            320, 1280,                      // Number of input and output channels
            35,                             // Weight import index
            var.block18_add,                // Input
            var.final_conv2d,               // Output
            par[0].final_par_conv2d);       // Weights

    batch_normalization(7,                  // Input/output size
            1280,                           // Number of channels
            52,                             // Parameter import index
            var.final_conv2d,               // Input
            var.final_conv2d_relu,          // Output
            par[0].final_par_conv2d_BN);    // Weights

    relu6(7,                                // Input/output size
            1280,                           // Number of channels
            var.final_conv2d_relu);         // Input/output

    avgpool(var.final_conv2d_relu,          // Input
            var.final_pooling);             // Output

    fully_connected(class,                  // Classification Layer Index
            var.final_pooling,              // Input
            predictions,                    // Output
            fc_w,                           // Weights
            fc_b);                          // Biases

    softmax(predictions,                    // Input/output
            class);                         // Length

    decode(predictions);                    // Input/output - Final prediction

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

void train(int c, float predictions[c], float fc_w[1280][c][2], float fc_b[c][2]){

    inference(c, predictions, fc_w, fc_b);
    printf("\n");

    // Final Block
    backprop_fc(c,                              // Classification Layer Parameter
            var.final_pooling,                  // Input
            predictions,                        // Output (recycled as backpropagated error from here on)
            label,                              // Correct prediction
            fc_w,                               // Weights and moving squared means
            fc_b);                              // Biases and moving squared means

    int i = 0;                  // Index for comparison
    if (frz == ++i) {return;}   // Freeze point - Fully Connected

    backprop_avrgpool(var.final_conv2d_relu,    // Input
            var.final_pooling);                 // Output

    backprop_relu6(7, 1280,                     // Size and depth
            var.final_conv2d_relu);             // Input/Output


    backprop_bn(7, 1280,                        // Size and Depth
            var.final_conv2d,                   // Input
            var.final_conv2d_relu,              // Output
            par[0].final_par_conv2d_BN,         // Parameters
            par[1].final_par_conv2d_BN,         // Parameter moving squared means
            52);                                // Import index

    backprop_conv2d(7, 7, 1, 320, 1280,         // Line size, column size, kernel size, input depth, output depth
            var.block18_add,                    // Input
            var.final_conv2d,                   // Output
            par[0].final_par_conv2d,            // Weights
            par[1].final_par_conv2d,            // Weight moving squared means
            1, 0, 35);                          // Stride, padding and import index

    if (frz == ++i) {return;}   // Freeze point - Block 18

    // Block 18
    backprop_S1(7,          // Input and output size
            160,            // Input depth
            32, 48, 16,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            960, 320,       // Expand and project factors
            var.block17_add,                                                // Input
            var.block18_expand,             var.block18_expand_relu,        // Expand output
            var.block18_depth,              var.block18_depth_relu,         // Depthwise output
            var.block18_project,            var.block18_add,                // Project output
            par[0].block18_par_expand,      par[1].block18_par_expand,      // Expand weights/MSM
            par[0].block18_par_expand_BN,   par[1].block18_par_expand_BN,   // Expand parameters/MSM
            par[0].block18_par_depth,       par[1].block18_par_depth,       // Depthwise weights/MSM
            par[0].block18_par_depth_BN,    par[1].block18_par_depth_BN,    // Depthwise parameters/MSM
            par[0].block18_par_project,     par[1].block18_par_project,     // Project weights/MSM
            par[0].block18_par_project_BN,  par[1].block18_par_project_BN); // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 17

    // Block 17
    backprop_S1(7,          // Input and output size
            160,            // Input depth
            30, 45, 15,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            960, 160,       // Expand and project factors
            var.block16_add,                                                // Input
            var.block17_expand,             var.block17_expand_relu,        // Expand output
            var.block17_depth,              var.block17_depth_relu,         // Depthwise output
            var.block17_project,            var.block17_add,                // Project output
            par[0].block17_par_expand,      par[1].block17_par_expand,      // Expand weights/MSM
            par[0].block17_par_expand_BN,   par[1].block17_par_expand_BN,   // Expand parameters/MSM
            par[0].block17_par_depth,       par[1].block17_par_depth,       // Depthwise weights/MSM
            par[0].block17_par_depth_BN,    par[1].block17_par_depth_BN,    // Depthwise parameters/MSM
            par[0].block17_par_project,     par[1].block17_par_project,     // Project weights/MSM
            par[0].block17_par_project_BN,  par[1].block17_par_project_BN); // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 16

    // Block 16
    backprop_S1(7,          // Input and output size
            160,            // Input depth
            28, 42, 14,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            960, 160,       // Expand and project factors
            var.block15_project_BN,                                         // Input
            var.block16_expand,             var.block16_expand_relu,        // Expand output
            var.block16_depth,              var.block16_depth_relu,         // Depthwise output
            var.block16_project,            var.block16_add,                // Project output
            par[0].block16_par_expand,      par[1].block16_par_expand,      // Expand weights/MSM
            par[0].block16_par_expand_BN,   par[1].block16_par_expand_BN,   // Expand parameters/MSM
            par[0].block16_par_depth,       par[1].block16_par_depth,       // Depthwise weights/MSM
            par[0].block16_par_depth_BN,    par[1].block16_par_depth_BN,    // Depthwise parameters/MSM
            par[0].block16_par_project,     par[1].block16_par_project,     // Project weights/MSM
            par[0].block16_par_project_BN,  par[1].block16_par_project_BN); // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 15

    // Block 15
    backprop_S2(14, 7,      // Input and output size / depthwise factor
            96, 160,        // input and output depth / project factor
            26, 39, 13,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            576,            // Expand factor
            var.block14_add,                                                // Input
            var.block15_expand,             var.block15_expand_relu,        // Expand output
            var.block15_depth,              var.block15_depth_relu,         // Depthwise output
            var.block15_project,            var.block15_project_BN,         // Project output
            par[0].block15_par_expand,      par[1].block15_par_expand,      // Expand weights/MSM
            par[0].block15_par_expand_BN,   par[1].block15_par_expand_BN,   // Expand parameters/MSM
            par[0].block15_par_depth,       par[1].block15_par_depth,       // Depthwise weights/MSM
            par[0].block15_par_depth_BN,    par[1].block15_par_depth_BN,    // Depthwise parameters/MSM
            par[0].block15_par_project,     par[1].block15_par_project,     // Project weights/MSM
            par[0].block15_par_project_BN,  par[1].block15_par_project_BN); // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 14

    // Block 14
    backprop_S1(14,         // Input and output size
            96,             // Input depth
            24, 36, 12,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            576, 96,        // Expand and project factors
            var.block13_add,                                                // Input
            var.block14_expand,             var.block14_expand_relu,        // Expand output
            var.block14_depth,              var.block14_depth_relu,         // Depthwise output
            var.block14_project,            var.block14_add,                // Project output
            par[0].block14_par_expand,      par[1].block14_par_expand,      // Expand weights/MSM
            par[0].block14_par_expand_BN,   par[1].block14_par_expand_BN,   // Expand parameters/MSM
            par[0].block14_par_depth,       par[1].block14_par_depth,       // Depthwise weights/MSM
            par[0].block14_par_depth_BN,    par[1].block14_par_depth_BN,    // Depthwise parameters/MSM
            par[0].block14_par_project,     par[1].block14_par_project,     // Project weights/MSM
            par[0].block14_par_project_BN,  par[1].block14_par_project_BN); // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 13

    // Block 13
    backprop_S1(14,         // Input and output size
            96,             // Input depth
            22, 33, 11,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            576, 96,        // Expand and project factors
            var.block12_add,                                                // Input
            var.block13_expand,             var.block13_expand_relu,        // Expand output
            var.block13_depth,              var.block13_depth_relu,         // Depthwise output
            var.block13_project,            var.block13_add,                // Project output
            par[0].block13_par_expand,      par[1].block13_par_expand,      // Expand weights/MSM
            par[0].block13_par_expand_BN,   par[1].block13_par_expand_BN,   // Expand parameters/MSM
            par[0].block13_par_depth,       par[1].block13_par_depth,       // Depthwise weights/MSM
            par[0].block13_par_depth_BN,    par[1].block13_par_depth_BN,    // Depthwise parameters/MSM
            par[0].block13_par_project,     par[1].block13_par_project,     // Project weights/MSM
            par[0].block13_par_project_BN,  par[1].block13_par_project_BN); // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 12

    // Block 12
    backprop_S1(14,         // Input and output size
            64,             // Input depth
            20, 30, 10,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            384, 96,        // Expand and project factors
            var.block11_add,                                                // Input
            var.block12_expand,             var.block12_expand_relu,        // Expand output
            var.block12_depth,              var.block12_depth_relu,         // Depthwise output
            var.block12_project,            var.block12_add,                // Project output
            par[0].block12_par_expand,      par[1].block12_par_expand,      // Expand weights/MSM
            par[0].block12_par_expand_BN,   par[1].block12_par_expand_BN,   // Expand parameters/MSM
            par[0].block12_par_depth,       par[1].block12_par_depth,       // Depthwise weights/MSM
            par[0].block12_par_depth_BN,    par[1].block12_par_depth_BN,    // Depthwise parameters/MSM
            par[0].block12_par_project,     par[1].block12_par_project,     // Project weights/MSM
            par[0].block12_par_project_BN,  par[1].block12_par_project_BN); // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 11

    // Block 11
    backprop_S1(14,         // Input and output size
            64,             // Input depth
            18, 27, 9,      // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            384, 64,        // Expand and project factors
            var.block10_add,                                                // Input
            var.block11_expand,             var.block11_expand_relu,        // Expand output
            var.block11_depth,              var.block11_depth_relu,         // Depthwise output
            var.block11_project,            var.block11_add,                // Project output
            par[0].block11_par_expand,      par[1].block11_par_expand,      // Expand weights/MSM
            par[0].block11_par_expand_BN,   par[1].block11_par_expand_BN,   // Expand parameters/MSM
            par[0].block11_par_depth,       par[1].block11_par_depth,       // Depthwise weights/MSM
            par[0].block11_par_depth_BN,    par[1].block11_par_depth_BN,    // Depthwise parameters/MSM
            par[0].block11_par_project,     par[1].block11_par_project,     // Project weights/MSM
            par[0].block11_par_project_BN,  par[1].block11_par_project_BN); // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 10

    // Block 10
    backprop_S1(14,         // Input and output size
            64,             // Input depth
            16, 24, 8,      // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            384, 64,        // Expand and project factors
            var.block9_add,                                                 // Input
            var.block10_expand,             var.block10_expand_relu,        // Expand output
            var.block10_depth,              var.block10_depth_relu,         // Depthwise output
            var.block10_project,            var.block10_add,                // Project output
            par[0].block10_par_expand,      par[1].block10_par_expand,      // Expand weights/MSM
            par[0].block10_par_expand_BN,   par[1].block10_par_expand_BN,   // Expand parameters/MSM
            par[0].block10_par_depth,       par[1].block10_par_depth,       // Depthwise weights/MSM
            par[0].block10_par_depth_BN,    par[1].block10_par_depth_BN,    // Depthwise parameters/MSM
            par[0].block10_par_project,     par[1].block10_par_project,     // Project weights/MSM
            par[0].block10_par_project_BN,  par[1].block10_par_project_BN); // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 9

    // Block 9
    backprop_S1(14,         // Input and output size
            64,             // Input depth
            14, 21, 7,      // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            384, 64,        // Expand and project factors
            var.block8_project_BN,                                          // Input
            var.block9_expand,              var.block9_expand_relu,         // Expand output
            var.block9_depth,               var.block9_depth_relu,          // Depthwise output
            var.block9_project,             var.block9_add,                 // Project output
            par[0].block9_par_expand,       par[1].block9_par_expand,       // Expand weights/MSM
            par[0].block9_par_expand_BN,    par[1].block9_par_expand_BN,    // Expand parameters/MSM
            par[0].block9_par_depth,        par[1].block9_par_depth,        // Depthwise weights/MSM
            par[0].block9_par_depth_BN,     par[1].block9_par_depth_BN,     // Depthwise parameters/MSM
            par[0].block9_par_project,      par[1].block9_par_project,      // Project weights/MSM
            par[0].block9_par_project_BN,   par[1].block9_par_project_BN);  // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 8

    // Block 8
    backprop_S2(28, 14,     // Input and output size / depthwise factor
            32, 64,         // input and output depth / project factor
            12, 18, 6,      // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            192,            // Expand factor
            var.block7_add,                                                 // Input
            var.block8_expand,              var.block8_expand_relu,         // Expand output
            var.block8_depth,               var.block8_depth_relu,          // Depthwise output
            var.block8_project,             var.block8_project_BN,          // Project output
            par[0].block8_par_expand,       par[1].block8_par_expand,       // Expand weights/MSM
            par[0].block8_par_expand_BN,    par[1].block8_par_expand_BN,    // Expand parameters/MSM
            par[0].block8_par_depth,        par[1].block8_par_depth,        // Depthwise weights/MSM
            par[0].block8_par_depth_BN,     par[1].block8_par_depth_BN,     // Depthwise parameters/MSM
            par[0].block8_par_project,      par[1].block8_par_project,      // Project weights/MSM
            par[0].block8_par_project_BN,   par[1].block8_par_project_BN);  // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 7

    // Block 7
    backprop_S1(28,         // Input and output size
            32,             // Input depth
            10, 15, 5,      // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            192, 32,        // Expand and project factors
            var.block6_add,                                                 // Input
            var.block7_expand,              var.block7_expand_relu,         // Expand output
            var.block7_depth,               var.block7_depth_relu,          // Depthwise output
            var.block7_project,             var.block7_add,                 // Project output
            par[0].block7_par_expand,       par[1].block7_par_expand,       // Expand weights/MSM
            par[0].block7_par_expand_BN,    par[1].block7_par_expand_BN,    // Expand parameters/MSM
            par[0].block7_par_depth,        par[1].block7_par_depth,        // Depthwise weights/MSM
            par[0].block7_par_depth_BN,     par[1].block7_par_depth_BN,     // Depthwise parameters/MSM
            par[0].block7_par_project,      par[1].block7_par_project,      // Project weights/MSM
            par[0].block7_par_project_BN,   par[1].block7_par_project_BN);  // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 6

    // Block 6
    backprop_S1(28,         // Input and output size
            32,             // Input depth
            8, 12, 4,       // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            192, 32,        // Expand and project factors
            var.block5_project_BN,                                          // Input
            var.block6_expand,              var.block6_expand_relu,         // Expand output
            var.block6_depth,               var.block6_depth_relu,          // Depthwise output
            var.block6_project,             var.block6_add,                 // Project output
            par[0].block6_par_expand,       par[1].block6_par_expand,       // Expand weights/MSM
            par[0].block6_par_expand_BN,    par[1].block6_par_expand_BN,    // Expand parameters/MSM
            par[0].block6_par_depth,        par[1].block6_par_depth,        // Depthwise weights/MSM
            par[0].block6_par_depth_BN,     par[1].block6_par_depth_BN,     // Depthwise parameters/MSM
            par[0].block6_par_project,      par[1].block6_par_project,      // Project weights/MSM
            par[0].block6_par_project_BN,   par[1].block6_par_project_BN);  // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 5

    // Block 5
    backprop_S2(56, 28,     // Input and output size / depthwise factor
            24, 32,         // input and output depth / project factor
            6, 9, 3,        // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            144,            // Expand factor
            var.block4_add,                                                 // Input
            var.block5_expand,              var.block5_expand_relu,         // Expand output
            var.block5_depth,               var.block5_depth_relu,          // Depthwise output
            var.block5_project,             var.block5_project_BN,          // Project output
            par[0].block5_par_expand,       par[1].block5_par_expand,       // Expand weights/MSM
            par[0].block5_par_expand_BN,    par[1].block5_par_expand_BN,    // Expand parameters/MSM
            par[0].block5_par_depth,        par[1].block5_par_depth,        // Depthwise weights/MSM
            par[0].block5_par_depth_BN,     par[1].block5_par_depth_BN,     // Depthwise parameters/MSM
            par[0].block5_par_project,      par[1].block5_par_project,      // Project weights/MSM
            par[0].block5_par_project_BN,   par[1].block5_par_project_BN);  // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 4

    // Block 4
    backprop_S1(56,         // Input and output size
            24,             // Input depth
            4, 6, 2,        // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            144, 24,        // Expand and project factors
            var.block3_project_BN,                                          // Input
            var.block4_expand,              var.block4_expand_relu,         // Expand output
            var.block4_depth,               var.block4_depth_relu,          // Depthwise output
            var.block4_project,             var.block4_add,                 // Project output
            par[0].block4_par_expand,       par[1].block4_par_expand,       // Expand weights/MSM
            par[0].block4_par_expand_BN,    par[1].block4_par_expand_BN,    // Expand parameters/MSM
            par[0].block4_par_depth,        par[1].block4_par_depth,        // Depthwise weights/MSM
            par[0].block4_par_depth_BN,     par[1].block4_par_depth_BN,     // Depthwise parameters/MSM
            par[0].block4_par_project,      par[1].block4_par_project,      // Project weights/MSM
            par[0].block4_par_project_BN,   par[1].block4_par_project_BN);  // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Block 3

    // Block 3
    backprop_S2(112, 56,    // Input and output size / depthwise factor
            16, 24,         // input and output depth / project factor
            2, 3, 1,        // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            96,             // Expand factor
            var.expanded_project_BN,                                        // Input
            var.block3_expand,              var.block3_expand_relu,         // Expand output
            var.block3_depth,               var.block3_depth_relu,          // Depthwise output
            var.block3_project,             var.block3_project_BN,          // Project output
            par[0].block3_par_expand,       par[1].block3_par_expand,       // Expand weights/MSM
            par[0].block3_par_expand_BN,    par[1].block3_par_expand_BN,    // Expand parameters/MSM
            par[0].block3_par_depth,        par[1].block3_par_depth,        // Depthwise weights/MSM
            par[0].block3_par_depth_BN,     par[1].block3_par_depth_BN,     // Depthwise parameters/MSM
            par[0].block3_par_project,      par[1].block3_par_project,      // Project weights/MSM
            par[0].block3_par_project_BN,   par[1].block3_par_project_BN);  // Project parameters/MSM

    if (frz == ++i) {return;}   // Freeze point - Expanded Block

    // Expanded Block
    backprop_bn(112, 16,                        // Size and depth
            var.expanded_project,               // Input
            var.expanded_project_BN,            // Output
            par[0].expanded_par_project_BN,     // Parameters
            par[1].expanded_par_project_BN,     // Parameter moving squared means
            3);                                 // Import index

    backprop_conv2d(112, 112, 1, 32, 16,        // Line size, column size, kernel size, input depth, output depth
            var.expanded_relu,                  // Input
            var.expanded_project,               // Output
            par[0].expanded_par_project,        // Weights
            par[1].expanded_par_project,        // Weight moving squared means
            1, 0, 2);                           // Stride, padding and import index

    backprop_relu6(112, 32,                     // Size and depth
            var.expanded_relu);                 // Input/output

    backprop_bn(112, 32,                        // Size and depth
            var.expanded_depth,                 // Input
            var.expanded_relu,                  // Output
            par[0].expanded_par_depth_BN,       // Parameters
            par[1].expanded_par_depth_BN,       // Parameter moving squared means
            2);                                 // Import index

    backprop_dw(112, 112, 3, 32,                // Input size, output size, kernel size, depth
            var.initial_relu,                   // Input
            var.expanded_depth,                 // Output
            par[0].expanded_par_depth,          // Weights
            par[1].expanded_par_depth,          // Weight moving squared means
            1, 1, 1);                           // Stride, padding and import index

    // Initial Block
    backprop_relu6(112, 32,                     // Size and depth
            var.initial_relu);                  // Input/output

    backprop_bn(112, 32,                        // Size and depth
            var.initial_conv2d,                 // Input
            var.initial_relu,                   // Output
            par[0].initial_par_BN,              // Parameters
            par[1].initial_par_BN,              // Parameter moving squared means
            1);                                 // Import index

    backprop_conv2d(224, 112, 3, 3, 32,         // Line size, column size, kernel size, input depth, output depth
            image,                              // Input
            var.initial_conv2d,                 // Output
            par[0].initial_par_conv2d,          // Weights
            par[1].initial_par_conv2d,          // Weight moving squared means
            2, 0, 1);                           // Stride, padding and import index
}

void test(int c, float predictions[c], float fc_w[1280][c][2], float fc_b[c][2], int n) {

    int NC = 0;                                                 // Number of correct predictions
    int (*conf)[c] = calloc(c, sizeof *conf);                   // Confusion matrix: actual (lines) x predicted (columns)
    float step = 0.02;                                          // Threshold Step
    int t_number = (1 - 0.5) / step;                            // Number of points
    int (*conf_arr)[2][t_number] = calloc(2, sizeof *conf_arr); // Confusion matrix array
    loss = 0;                                                   // Reset loss

    // Test
    for (int i = 0; i < n; ++i) {
        printf("Test/Validation Image %d\n", i);
        import_image(i, 1);
        inference(class, predictions, fc_w, fc_b);

        // Add to Correct Predictions
        if (inf_idx == label)
            NC++;

        // Add to Confusion Matrix
        conf[label][inf_idx]++;

        // Add to Confusion Matrix Array on Thresholds
        if (c == 2) {
            for (int t_count = 0; t_count < t_number; ++t_count) {
                float th = 0.5 + (t_count * step);  // Current threshold
                int th_idx;                         // Index within threshold

                // Calculate th_idx
                if (inf_pred >= th)
                    th_idx = inf_idx;
                else
                    th_idx = !inf_idx;

                conf_arr[label][th_idx][t_count]++;
            }
        }

        // Add to cross-entropy loss
        loss += -nat_log(inf_correct);   // Maybe toss standard weight decay here too
    }

    // Classification Accuracy
    float acc = (float)NC/n;
    printf("\nClassification Accuracy = %.2f%%\n", acc*100);

    // Confusion Matrix
    printf("\nConfusion Matrix = \n");
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < c; ++j) {
            printf("%d   ",conf[i][j]);
        }
        printf("\n");
    }

    // Precision, Recall and F1 Score (currently only for class = 2)
    if (c == 2) {
        float precision = (float)conf[1][1] / (conf[1][1] + conf[0][1]);    // Precision = TP / (TP + FP)
        float recall = (float)conf[1][1] / (conf[1][1] + conf[1][0]);       // Recall = TP / (TP + FN)      aka Sensitivity
        float F1 = 2.0 / ( (1/precision) + (1/recall) );                    // F1 Score
        printf("\nPrecision = %f\nRecall = %f\nF1 Score = %f\n", precision, recall, F1);

    // Receiver Operating Characteristic Curve and ROC Area Under Curve (only for class = 2)
        float (*ROC)[t_number + 1] = calloc(2, sizeof *ROC);    // ROC Curve: 0 -> FPR / 1 -> TPR
        float area = 0;                                         // Area Under Curve
        for (int t_count = 0; t_count < t_number; ++t_count) {
            ROC[1][t_count] = (float)conf_arr[1][1][t_count] / (conf_arr[1][0][t_count] + conf_arr[1][1][t_count]);    // TPR = TP / (FN + TP)     aka Sensitivity
            ROC[0][t_count] = (float)conf_arr[0][1][t_count] / (conf_arr[0][0][t_count] + conf_arr[0][1][t_count]);    // FPR = FP / (TN + FP)     aka 1 - Specificity
            if (t_count != 0) {
                area += absolute( ( (ROC[1][t_count] + ROC[1][t_count-1]) / 2) * (ROC[0][t_count] - ROC[0][t_count-1]) );  // Area Under Curve (trapezoid approximation)
            }
        }
        ROC[1][t_number] = 1;
        ROC[0][t_number] = 1;
        area += absolute( ( (1 + ROC[1][t_number-1]) / 2) * (1 - ROC[0][t_number-1]) );

        /* Print confusion array matrixes
        printf("\nConfusion Array = \n");
        for (int t_count = 0; t_count < t_number; ++t_count) {
            printf("\nMatrix at threshold %.2f:\n", 0.5 + (t_count * step));
            for (int i = 0; i < c; ++i) {
                for (int j = 0; j < c; ++j) {
                    printf("%d   ",conf_arr[i][j][t_count]);
                }
                printf("\n");
            }
        }
        */

        // Receiver Operating Characteristic AUC
        printf("\nArea Under ROC Curve = %.3f\n", area);

        // Export to gnuplot
        char name[15]; // maximum number of characters is "roc_dataxx.txt" = 14
        sprintf(name,"roc_data%d.txt",epoch_count);
        FILE *fptr;
        fptr = fopen(name, "w");
        if (fptr == NULL) {
            perror("fopen()");
            exit(EXIT_FAILURE);
        }

        for (int t_count = 0; t_count < t_number+1; ++t_count)
            fprintf(fptr, "%f %f\n", ROC[0][t_count], ROC[1][t_count]);

        fclose (fptr);
        free(ROC);
    }
    free(conf);
    free(conf_arr);

    // Cross-entropy Loss
    printf("\nCross-entropy Loss = %f", loss);
    loss_plot(epoch_count);

}

void transfer() {

    int i = 1;                  // Index for comparison

    // Fully Connected
    fillRandom("fc_w.csv", class*1280);
    fillRandom("fc_b.csv", class);

    // Final Block
    if (frz >= ++i) {
        fillRandom("param52.csv",   sizeof(par->final_par_conv2d_BN)/4);
        fillRandom("weights35.csv", sizeof(par->final_par_conv2d)/4);           // Problem is in Conv2d parameter update
    }

    // Block 18
    if (frz >= ++i) {
        fillRandom("param51.csv",    sizeof(par->block18_par_project_BN)/4);
        fillRandom("weights34.csv",  sizeof(par->block18_par_project)/4);
        fillRandom("param50.csv",    sizeof(par->block18_par_depth_BN)/4);
        fillRandom("dweights17.csv", sizeof(par->block18_par_depth)/4);
        fillRandom("param49.csv",    sizeof(par->block18_par_expand_BN)/4);
        fillRandom("weights33.csv",  sizeof(par->block18_par_expand)/4);
    }

    // Block 17
    if (frz >= ++i) {
        fillRandom("param48.csv",    sizeof(par->block17_par_project_BN)/4);
        fillRandom("weights32.csv",  sizeof(par->block17_par_project)/4);
        fillRandom("param47.csv",    sizeof(par->block17_par_depth_BN)/4);
        fillRandom("dweights16.csv", sizeof(par->block17_par_depth)/4);
        fillRandom("param46.csv",    sizeof(par->block17_par_expand_BN)/4);
        fillRandom("weights31.csv",  sizeof(par->block17_par_expand)/4);
    }

    // Block 16
    if (frz >= ++i) {
        fillRandom("param45.csv",    sizeof(par->block16_par_project_BN)/4);
        fillRandom("weights30.csv",  sizeof(par->block16_par_project)/4);
        fillRandom("param44.csv",    sizeof(par->block16_par_depth_BN)/4);
        fillRandom("dweights15.csv", sizeof(par->block16_par_depth)/4);
        fillRandom("param43.csv",    sizeof(par->block16_par_expand_BN)/4);
        fillRandom("weights29.csv",  sizeof(par->block16_par_expand)/4);
    }

    // Block 15
    if (frz >= ++i) {
        fillRandom("param42.csv",    sizeof(par->block15_par_project_BN)/4);
        fillRandom("weights28.csv",  sizeof(par->block15_par_project)/4);
        fillRandom("param41.csv",    sizeof(par->block15_par_depth_BN)/4);
        fillRandom("dweights14.csv", sizeof(par->block15_par_depth)/4);
        fillRandom("param40.csv",    sizeof(par->block15_par_expand_BN)/4);
        fillRandom("weights27.csv",  sizeof(par->block15_par_expand)/4);
    }

    // Block 14
    if (frz >= ++i) {
        fillRandom("param39.csv",    sizeof(par->block14_par_project_BN)/4);
        fillRandom("weights26.csv",  sizeof(par->block14_par_project)/4);
        fillRandom("param38.csv",    sizeof(par->block14_par_depth_BN)/4);
        fillRandom("dweights13.csv", sizeof(par->block14_par_depth)/4);
        fillRandom("param37.csv",    sizeof(par->block14_par_expand_BN)/4);
        fillRandom("weights25.csv",  sizeof(par->block14_par_expand)/4);
    }

    // Block 13
    if (frz >= ++i) {
        fillRandom("param36.csv",    sizeof(par->block13_par_project_BN)/4);
        fillRandom("weights24.csv",  sizeof(par->block13_par_project)/4);
        fillRandom("param35.csv",    sizeof(par->block13_par_depth_BN)/4);
        fillRandom("dweights12.csv", sizeof(par->block13_par_depth)/4);
        fillRandom("param34.csv",    sizeof(par->block13_par_expand_BN)/4);
        fillRandom("weights23.csv",  sizeof(par->block13_par_expand)/4);
    }

    // Block 12
    if (frz >= ++i) {
        fillRandom("param33.csv",    sizeof(par->block12_par_project_BN)/4);
        fillRandom("weights22.csv",  sizeof(par->block12_par_project)/4);
        fillRandom("param32.csv",    sizeof(par->block12_par_depth_BN)/4);
        fillRandom("dweights11.csv", sizeof(par->block12_par_depth)/4);
        fillRandom("param31.csv",    sizeof(par->block12_par_expand_BN)/4);
        fillRandom("weights21.csv",  sizeof(par->block12_par_expand)/4);
    }

    // Block 11
    if (frz >= ++i) {
        fillRandom("param30.csv",    sizeof(par->block11_par_project_BN)/4);
        fillRandom("weights20.csv",  sizeof(par->block11_par_project)/4);
        fillRandom("param29.csv",    sizeof(par->block11_par_depth_BN)/4);
        fillRandom("dweights10.csv", sizeof(par->block11_par_depth)/4);
        fillRandom("param28.csv",    sizeof(par->block11_par_expand_BN)/4);
        fillRandom("weights19.csv",  sizeof(par->block11_par_expand)/4);
    }

    // Block 10
    if (frz >= ++i) {
        fillRandom("param27.csv",    sizeof(par->block10_par_project_BN)/4);
        fillRandom("weights18.csv",  sizeof(par->block10_par_project)/4);
        fillRandom("param26.csv",    sizeof(par->block10_par_depth_BN)/4);
        fillRandom("dweights9.csv",  sizeof(par->block10_par_depth)/4);
        fillRandom("param25.csv",    sizeof(par->block10_par_expand_BN)/4);
        fillRandom("weights17.csv",  sizeof(par->block10_par_expand)/4);
    }

    // Block 9
    if (frz >= ++i) {
        fillRandom("param24.csv",    sizeof(par->block9_par_project_BN)/4);
        fillRandom("weights16.csv",  sizeof(par->block9_par_project)/4);
        fillRandom("param23.csv",    sizeof(par->block9_par_depth_BN)/4);
        fillRandom("dweights8.csv",  sizeof(par->block9_par_depth)/4);
        fillRandom("param22.csv",    sizeof(par->block9_par_expand_BN)/4);
        fillRandom("weights15.csv",  sizeof(par->block9_par_expand)/4);
    }

    // Block 8
    if (frz >= ++i) {
        fillRandom("param21.csv",    sizeof(par->block8_par_project_BN)/4);
        fillRandom("weights14.csv",  sizeof(par->block8_par_project)/4);
        fillRandom("param20.csv",    sizeof(par->block8_par_depth_BN)/4);
        fillRandom("dweights7.csv",  sizeof(par->block8_par_depth)/4);
        fillRandom("param19.csv",    sizeof(par->block8_par_expand_BN)/4);
        fillRandom("weights13.csv",  sizeof(par->block8_par_expand)/4);
    }

    // Block 7
    if (frz >= ++i) {
        fillRandom("param18.csv",    sizeof(par->block7_par_project_BN)/4);
        fillRandom("weights12.csv",  sizeof(par->block7_par_project)/4);
        fillRandom("param17.csv",    sizeof(par->block7_par_depth_BN)/4);
        fillRandom("dweights6.csv",  sizeof(par->block7_par_depth)/4);
        fillRandom("param16.csv",    sizeof(par->block7_par_expand_BN)/4);
        fillRandom("weights11.csv",  sizeof(par->block7_par_expand)/4);
    }

    // Block 6
    if (frz >= ++i) {
        fillRandom("param15.csv",    sizeof(par->block6_par_project_BN)/4);
        fillRandom("weights10.csv",  sizeof(par->block6_par_project)/4);
        fillRandom("param14.csv",    sizeof(par->block6_par_depth_BN)/4);
        fillRandom("dweights5.csv",  sizeof(par->block6_par_depth)/4);
        fillRandom("param13.csv",    sizeof(par->block6_par_expand_BN)/4);
        fillRandom("weights9.csv",   sizeof(par->block6_par_expand)/4);
    }

    // Block 5
    if (frz >= ++i) {
        fillRandom("param12.csv",    sizeof(par->block5_par_project_BN)/4);
        fillRandom("weights8.csv",   sizeof(par->block5_par_project)/4);
        fillRandom("param11.csv",    sizeof(par->block5_par_depth_BN)/4);
        fillRandom("dweights4.csv",  sizeof(par->block5_par_depth)/4);
        fillRandom("param10.csv",    sizeof(par->block5_par_expand_BN)/4);
        fillRandom("weights7.csv",   sizeof(par->block5_par_expand)/4);
    }

    // Block 4
    if (frz >= ++i) {
        fillRandom("param9.csv",     sizeof(par->block4_par_project_BN)/4);
        fillRandom("weights6.csv",   sizeof(par->block4_par_project)/4);
        fillRandom("param8.csv",     sizeof(par->block4_par_depth_BN)/4);
        fillRandom("dweights3.csv",  sizeof(par->block4_par_depth)/4);
        fillRandom("param7.csv",     sizeof(par->block4_par_expand_BN)/4);
        fillRandom("weights5.csv",   sizeof(par->block4_par_expand)/4);
    }

    // Block 3
    if (frz >= ++i) {
        fillRandom("param6.csv",     sizeof(par->block3_par_project_BN)/4);
        fillRandom("weights4.csv",   sizeof(par->block3_par_project)/4);
        fillRandom("param5.csv",     sizeof(par->block3_par_depth_BN)/4);
        fillRandom("dweights2.csv",  sizeof(par->block3_par_depth)/4);
        fillRandom("param4.csv",     sizeof(par->block3_par_expand_BN)/4);
        fillRandom("weights3.csv",   sizeof(par->block3_par_expand)/4);
    }

    // Expanded Block
    if (frz >= ++i) {
        fillRandom("param3.csv",     sizeof(par->expanded_par_project_BN)/4);
        fillRandom("weights2.csv",   sizeof(par->expanded_par_project)/4);
        fillRandom("param2.csv",     sizeof(par->block3_par_depth_BN)/4);
        fillRandom("dweights1.csv",  sizeof(par->expanded_par_depth)/4);

    // Initial Block
        fillRandom("param1.csv",     sizeof(par->initial_par_BN)/4);
        fillRandom("weights1.csv",   sizeof(par->initial_par_conv2d)/4);
    }

    clock_t begin = clock();

    // Train
    printf("\n");
    float predictions[class];
    float (*fc_w)[class][2] = calloc(1280, sizeof *fc_w);
    float (*fc_b)[2] = calloc(class, sizeof *fc_b);

    for (epoch_count = 0; epoch_count < n_epoch; ++epoch_count) {
        for (int i = 0; i < n_img; ++i) {
            printf("Epoch %d Image %d\n",epoch_count + 1, i + 1);
            import_image(i, 0);
            train(class, predictions, fc_w, fc_b);
            printf("\n");
        }
        if (n_val)
            test(class, predictions, fc_w, fc_b, n_val);
    }

    clock_t end = clock();
    double train_time = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\nTrain time with validation: %fs\n", train_time);

    free(fc_w);
    free(fc_b);
}

