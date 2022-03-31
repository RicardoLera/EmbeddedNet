#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "blocks.h"
#include "layers.h"
#include "operations.h"
#include "data_manip.h"
#include "var.h"

void inference(int c, float predictions[c], float fc_w[1280][c], float fc_b[c]) {

    printf("\nInferring...\n");

    // Initial Block
    //printf("Entering Initial Block: 224x224x3\n");


    conv2d(224, 112, 3,                         // Input size, output size, kernel size
            2,                                  // Stride
            0,                                  // Padding
            3, 32,                              // Number of input and output channels
            1,                                  // Weight import index
            image,                              // Input
            var.initial_conv2d,                 // Output
            par.initial_par_conv2d);            // Weights


/* Export test
    printf("=%.20f\n", -0.17165923 * -0.33373192);

    printf("weight[0][0][0][0] = %.8e\n", par.initial_par_conv2d[0][0][0][0]);
    printf("weight[0][0][0][1] = %.8e\n", par.initial_par_conv2d[1][0][0][0]);
    printf("weight[0][0][1][0] = %.8e\n", par.initial_par_conv2d[0][0][0][1]);
    printf("weight Last = %.8e\n", par.initial_par_conv2d[31][2][2][2]);

    FILE *fp;
    fp = fopen("../data/output.txt", "w");

    for (int i = 0; i < 112; ++i) {
        for (int j = 0; j < 112; ++j) {
            fprintf(fp, "%.8e, ", var.initial_conv2d[i][j][0]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    exit(0);
*/


    batch_normalization(112,                    // Input/output size
            32,                                 // Number of channels
            1,                                  // Parameter import index
            var.initial_conv2d,                 // Input
            var.initial_relu,                   // Output
            par.initial_par_BN);                // Parameters

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
            par.expanded_par_depth);            // Weights

    batch_normalization(112,                    // Input/output size
            32,                                 // Number of channels
            2,                                  // Parameter import index
            var.expanded_depth,                 // Input
            var.expanded_relu,                  // Output
            par.expanded_par_depth_BN);         // Parameters

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
            par.expanded_par_project);          // Weights

    batch_normalization(112,                    // Input/output size
            16,                                 // Number of channels
            3,                                  // Parameter import index
            var.expanded_project,               // Input
            var.expanded_project_BN,            // Output
            par.expanded_par_project_BN);       // Parameters


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
            par.block3_par_expand,   par.block3_par_expand_BN,    // Expand parameters
            par.block3_par_depth,    par.block3_par_depth_BN,     // Depthwise parameters
            par.block3_par_project,  par.block3_par_project_BN);  // Project parameters


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
            par.block4_par_expand,   par.block4_par_expand_BN,    // Expand parameters
            par.block4_par_depth,    par.block4_par_depth_BN,     // Depthwise parameters
            par.block4_par_project,  par.block4_par_project_BN);  // Project parameters


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
            par.block5_par_expand,   par.block5_par_expand_BN,    // Expand parameters
            par.block5_par_depth,    par.block5_par_depth_BN,     // Depthwise parameters
            par.block5_par_project,  par.block5_par_project_BN);  // Project parameters


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
            par.block6_par_expand,   par.block6_par_expand_BN,    // Expand parameters
            par.block6_par_depth,    par.block6_par_depth_BN,     // Depthwise parameters
            par.block6_par_project,  par.block6_par_project_BN);  // Project parameters


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
            par.block7_par_expand,   par.block7_par_expand_BN,    // Expand parameters
            par.block7_par_depth,    par.block7_par_depth_BN,     // Depthwise parameters
            par.block7_par_project,  par.block7_par_project_BN);  // Project parameters


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
            par.block8_par_expand,   par.block8_par_expand_BN,    // Expand parameters
            par.block8_par_depth,    par.block8_par_depth_BN,     // Depthwise parameters
            par.block8_par_project,  par.block8_par_project_BN);  // Project parameters


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
            par.block9_par_expand,   par.block9_par_expand_BN,    // Expand parameters
            par.block9_par_depth,    par.block9_par_depth_BN,     // Depthwise parameters
            par.block9_par_project,  par.block9_par_project_BN);  // Project parameters


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
            par.block10_par_expand,  par.block10_par_expand_BN,   // Expand parameters
            par.block10_par_depth,   par.block10_par_depth_BN,    // Depthwise parameters
            par.block10_par_project, par.block10_par_project_BN); // Project parameters


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
            par.block11_par_expand,  par.block11_par_expand_BN,   // Expand parameters
            par.block11_par_depth,   par.block11_par_depth_BN,    // Depthwise parameters
            par.block11_par_project, par.block11_par_project_BN); // Project parameters


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
            par.block12_par_expand,  par.block12_par_expand_BN,   // Expand parameters
            par.block12_par_depth,   par.block12_par_depth_BN,    // Depthwise parameters
            par.block12_par_project, par.block12_par_project_BN); // Project parameters


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
            par.block13_par_expand,  par.block13_par_expand_BN,   // Expand parameters
            par.block13_par_depth,   par.block13_par_depth_BN,    // Depthwise parameters
            par.block13_par_project, par.block13_par_project_BN); // Project parameters


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
            par.block14_par_expand,  par.block14_par_expand_BN,   // Expand parameters
            par.block14_par_depth,   par.block14_par_depth_BN,    // Depthwise parameters
            par.block14_par_project, par.block14_par_project_BN); // Project parameters


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
            par.block15_par_expand,  par.block15_par_expand_BN,   // Expand parameters
            par.block15_par_depth,   par.block15_par_depth_BN,    // Depthwise parameters
            par.block15_par_project, par.block15_par_project_BN); // Project parameters


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
            par.block16_par_expand,  par.block16_par_expand_BN,   // Expand parameters
            par.block16_par_depth,   par.block16_par_depth_BN,    // Depthwise parameters
            par.block16_par_project, par.block16_par_project_BN); // Project parameters


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
            par.block17_par_expand,  par.block17_par_expand_BN,   // Expand parameters
            par.block17_par_depth,   par.block17_par_depth_BN,    // Depthwise parameters
            par.block17_par_project, par.block17_par_project_BN); // Project parameters


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
            par.block18_par_expand,  par.block18_par_expand_BN,   // Expand parameters
            par.block18_par_depth,   par.block18_par_depth_BN,    // Depthwise parameters
            par.block18_par_project, par.block18_par_project_BN); // Project parameters


    // Final Block
    //printf("Entering Final Block: 7x7x320\n");

    conv2d(7, 7, 1,                         // Input size, output size, kernel size
            1,                              // Stride
            0,                              // Padding
            320, 1280,                      // Number of input and output channels
            35,                             // Weight import index
            var.block18_add,                // Input
            var.final_conv2d,               // Output
            par.final_par_conv2d);       // Weights

    batch_normalization(7,                  // Input/output size
            1280,                           // Number of channels
            52,                             // Parameter import index
            var.final_conv2d,               // Input
            var.final_conv2d_relu,          // Output
            par.final_par_conv2d_BN);    // Weights

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
}

void train(int c, float predictions[c], float fc_w[1280][c], float fc_b[c]){

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
    if (frz == ++i) return;        // Freeze point - Fully Connected

    backprop_avrgpool(var.final_conv2d_relu,    // Input
            var.final_pooling);                 // Output

    backprop_relu6(7, 1280,                     // Size and depth
            var.final_conv2d_relu);             // Input/Output


    backprop_bn(7, 1280,                        // Size and Depth
            var.final_conv2d,                   // Input
            var.final_conv2d_relu,              // Output
            par.final_par_conv2d_BN,            // Parameters
            52);                                // Import index

    backprop_conv2d(7, 7, 1, 320, 1280,         // Line size, column size, kernel size, input depth, output depth
            var.block18_add,                    // Input
            var.final_conv2d,                   // Output
            par.final_par_conv2d,               // Weights
            1, 0, 35);                          // Stride, padding and import index

    if (frz == ++i) return;   // Freeze point - Block 18

    // Block 18
    backprop_S1(7,          // Input and output size
            160,            // Input depth
            32, 48, 16,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            960, 320,       // Expand and project factors
            var.block17_add,                                                // Input
            var.block18_expand,             var.block18_expand_relu,        // Expand output
            var.block18_depth,              var.block18_depth_relu,         // Depthwise output
            var.block18_project,            var.block18_add,                // Project output
            par.block18_par_expand,         par.block18_par_expand_BN,      // Expand parameters
            par.block18_par_depth,          par.block18_par_depth_BN,       // Depthwise parameters
            par.block18_par_project,        par.block18_par_project_BN);    // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 17

    // Block 17
    backprop_S1(7,          // Input and output size
            160,            // Input depth
            30, 45, 15,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            960, 160,       // Expand and project factors
            var.block16_add,                                                // Input
            var.block17_expand,             var.block17_expand_relu,        // Expand output
            var.block17_depth,              var.block17_depth_relu,         // Depthwise output
            var.block17_project,            var.block17_add,                // Project output
            par.block17_par_expand,         par.block17_par_expand_BN,      // Expand parameters
            par.block17_par_depth,          par.block17_par_depth_BN,       // Depthwise parameters
            par.block17_par_project,        par.block17_par_project_BN);    // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 16

    // Block 16
    backprop_S1(7,          // Input and output size
            160,            // Input depth
            28, 42, 14,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            960, 160,       // Expand and project factors
            var.block15_project_BN,                                         // Input
            var.block16_expand,             var.block16_expand_relu,        // Expand output
            var.block16_depth,              var.block16_depth_relu,         // Depthwise output
            var.block16_project,            var.block16_add,                // Project output
            par.block16_par_expand,         par.block16_par_expand_BN,      // Expand parameters
            par.block16_par_depth,          par.block16_par_depth_BN,       // Depthwise parameters
            par.block16_par_project,        par.block16_par_project_BN);    // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 15

    // Block 15
    backprop_S2(14, 7,      // Input and output size / depthwise factor
            96, 160,        // input and output depth / project factor
            26, 39, 13,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            576,            // Expand factor
            var.block14_add,                                                // Input
            var.block15_expand,             var.block15_expand_relu,        // Expand output
            var.block15_depth,              var.block15_depth_relu,         // Depthwise output
            var.block15_project,            var.block15_project_BN,         // Project output
            par.block15_par_expand,         par.block15_par_expand_BN,      // Expand parameters
            par.block15_par_depth,          par.block15_par_depth_BN,       // Depthwise parameters
            par.block15_par_project,        par.block15_par_project_BN);    // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 14

    // Block 14
    backprop_S1(14,         // Input and output size
            96,             // Input depth
            24, 36, 12,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            576, 96,        // Expand and project factors
            var.block13_add,                                                // Input
            var.block14_expand,             var.block14_expand_relu,        // Expand output
            var.block14_depth,              var.block14_depth_relu,         // Depthwise output
            var.block14_project,            var.block14_add,                // Project output
            par.block14_par_expand,         par.block14_par_expand_BN,      // Expand parameters
            par.block14_par_depth,          par.block14_par_depth_BN,       // Depthwise parameters
            par.block14_par_project,        par.block14_par_project_BN);    // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 13

    // Block 13
    backprop_S1(14,         // Input and output size
            96,             // Input depth
            22, 33, 11,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            576, 96,        // Expand and project factors
            var.block12_add,                                                // Input
            var.block13_expand,             var.block13_expand_relu,        // Expand output
            var.block13_depth,              var.block13_depth_relu,         // Depthwise output
            var.block13_project,            var.block13_add,                // Project output
            par.block13_par_expand,         par.block13_par_expand_BN,      // Expand parameters
            par.block13_par_depth,          par.block13_par_depth_BN,       // Depthwise parameters
            par.block13_par_project,        par.block13_par_project_BN);    // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 12

    // Block 12
    backprop_S1(14,         // Input and output size
            64,             // Input depth
            20, 30, 10,     // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            384, 96,        // Expand and project factors
            var.block11_add,                                                // Input
            var.block12_expand,             var.block12_expand_relu,        // Expand output
            var.block12_depth,              var.block12_depth_relu,         // Depthwise output
            var.block12_project,            var.block12_add,                // Project output
            par.block12_par_expand,         par.block12_par_expand_BN,      // Expand parameters
            par.block12_par_depth,          par.block12_par_depth_BN,       // Depthwise parameters
            par.block12_par_project,        par.block12_par_project_BN);    // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 11

    // Block 11
    backprop_S1(14,         // Input and output size
            64,             // Input depth
            18, 27, 9,      // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            384, 64,        // Expand and project factors
            var.block10_add,                                                // Input
            var.block11_expand,             var.block11_expand_relu,        // Expand output
            var.block11_depth,              var.block11_depth_relu,         // Depthwise output
            var.block11_project,            var.block11_add,                // Project output
            par.block11_par_expand,         par.block11_par_expand_BN,      // Expand parameters
            par.block11_par_depth,          par.block11_par_depth_BN,       // Depthwise parameters
            par.block11_par_project,        par.block11_par_project_BN);    // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 10

    // Block 10
    backprop_S1(14,         // Input and output size
            64,             // Input depth
            16, 24, 8,      // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            384, 64,        // Expand and project factors
            var.block9_add,                                                 // Input
            var.block10_expand,             var.block10_expand_relu,        // Expand output
            var.block10_depth,              var.block10_depth_relu,         // Depthwise output
            var.block10_project,            var.block10_add,                // Project output
            par.block10_par_expand,         par.block10_par_expand_BN,      // Expand parameters
            par.block10_par_depth,          par.block10_par_depth_BN,       // Depthwise parameters
            par.block10_par_project,        par.block10_par_project_BN);    // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 9

    // Block 9
    backprop_S1(14,         // Input and output size
            64,             // Input depth
            14, 21, 7,      // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            384, 64,        // Expand and project factors
            var.block8_project_BN,                                          // Input
            var.block9_expand,              var.block9_expand_relu,         // Expand output
            var.block9_depth,               var.block9_depth_relu,          // Depthwise output
            var.block9_project,             var.block9_add,                 // Project output
            par.block9_par_expand,          par.block9_par_expand_BN,       // Expand parameters
            par.block9_par_depth,           par.block9_par_depth_BN,        // Depthwise parameters
            par.block9_par_project,         par.block9_par_project_BN);     // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 8

    // Block 8
    backprop_S2(28, 14,     // Input and output size / depthwise factor
            32, 64,         // input and output depth / project factor
            12, 18, 6,      // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            192,            // Expand factor
            var.block7_add,                                                 // Input
            var.block8_expand,              var.block8_expand_relu,         // Expand output
            var.block8_depth,               var.block8_depth_relu,          // Depthwise output
            var.block8_project,             var.block8_project_BN,          // Project output
            par.block8_par_expand,          par.block8_par_expand_BN,       // Expand parameters
            par.block8_par_depth,           par.block8_par_depth_BN,        // Depthwise parameters
            par.block8_par_project,         par.block8_par_project_BN);     // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 7

    // Block 7
    backprop_S1(28,         // Input and output size
            32,             // Input depth
            10, 15, 5,      // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            192, 32,        // Expand and project factors
            var.block6_add,                                                 // Input
            var.block7_expand,              var.block7_expand_relu,         // Expand output
            var.block7_depth,               var.block7_depth_relu,          // Depthwise output
            var.block7_project,             var.block7_add,                 // Project output
            par.block7_par_expand,          par.block7_par_expand_BN,       // Expand parameters
            par.block7_par_depth,           par.block7_par_depth_BN,        // Depthwise parameters
            par.block7_par_project,         par.block7_par_project_BN);     // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 6

    // Block 6
    backprop_S1(28,         // Input and output size
            32,             // Input depth
            8, 12, 4,       // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            192, 32,        // Expand and project factors
            var.block5_project_BN,                                          // Input
            var.block6_expand,              var.block6_expand_relu,         // Expand output
            var.block6_depth,               var.block6_depth_relu,          // Depthwise output
            var.block6_project,             var.block6_add,                 // Project output
            par.block6_par_expand,          par.block6_par_expand_BN,       // Expand parameters
            par.block6_par_depth,           par.block6_par_depth_BN,        // Depthwise parameters
            par.block6_par_project,         par.block6_par_project_BN);     // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 5

    // Block 5
    backprop_S2(56, 28,     // Input and output size / depthwise factor
            24, 32,         // input and output depth / project factor
            6, 9, 3,        // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            144,            // Expand factor
            var.block4_add,                                                 // Input
            var.block5_expand,              var.block5_expand_relu,         // Expand output
            var.block5_depth,               var.block5_depth_relu,          // Depthwise output
            var.block5_project,             var.block5_project_BN,          // Project output
            par.block5_par_expand,          par.block5_par_expand_BN,       // Expand parameters
            par.block5_par_depth,           par.block5_par_depth_BN,        // Depthwise parameters
            par.block5_par_project,         par.block5_par_project_BN);     // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 4

    // Block 4
    backprop_S1(56,         // Input and output size
            24,             // Input depth
            4, 6, 2,        // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            144, 24,        // Expand and project factors
            var.block3_project_BN,                                          // Input
            var.block4_expand,              var.block4_expand_relu,         // Expand output
            var.block4_depth,               var.block4_depth_relu,          // Depthwise output
            var.block4_project,             var.block4_add,                 // Project output
            par.block4_par_expand,          par.block4_par_expand_BN,       // Expand parameters
            par.block4_par_depth,           par.block4_par_depth_BN,        // Depthwise parameters
            par.block4_par_project,         par.block4_par_project_BN);     // Project parameters

    if (frz == ++i) return;   // Freeze point - Block 3

    // Block 3
    backprop_S2(112, 56,    // Input and output size / depthwise factor
            16, 24,         // input and output depth / project factor
            2, 3, 1,        // 2D Convolution, Batch Normalization and Depthwise CURRENT parameter import indexes
            96,             // Expand factor
            var.expanded_project_BN,                                        // Input
            var.block3_expand,              var.block3_expand_relu,         // Expand output
            var.block3_depth,               var.block3_depth_relu,          // Depthwise output
            var.block3_project,             var.block3_project_BN,          // Project output
            par.block3_par_expand,          par.block3_par_expand_BN,       // Expand parameters
            par.block3_par_depth,           par.block3_par_depth_BN,        // Depthwise parameters
            par.block3_par_project,         par.block3_par_project_BN);     // Project parameters

    if (frz == ++i) return;   // Freeze point - Expanded Block

    // Expanded Block
    backprop_bn(112, 16,                        // Size and depth
            var.expanded_project,               // Input
            var.expanded_project_BN,            // Output
            par.expanded_par_project_BN,        // Parameters
            3);                                 // Import index

    backprop_conv2d(112, 112, 1, 32, 16,        // Line size, column size, kernel size, input depth, output depth
            var.expanded_relu,                  // Input
            var.expanded_project,               // Output
            par.expanded_par_project,           // Weights
            1, 0, 2);                           // Stride, padding and import index

    backprop_relu6(112, 32,                     // Size and depth
            var.expanded_relu);                 // Input/output

    backprop_bn(112, 32,                        // Size and depth
            var.expanded_depth,                 // Input
            var.expanded_relu,                  // Output
            par.expanded_par_depth_BN,          // Parameters
            2);                                 // Import index

    backprop_dw(112, 112, 3, 32,                // Input size, output size, kernel size, depth
            var.initial_relu,                   // Input
            var.expanded_depth,                 // Output
            par.expanded_par_depth,             // Weights
            1, 1, 1);                           // Stride, padding and import index

    // Initial Block
    backprop_relu6(112, 32,                     // Size and depth
            var.initial_relu);                  // Input/output

    backprop_bn(112, 32,                        // Size and depth
            var.initial_conv2d,                 // Input
            var.initial_relu,                   // Output
            par.initial_par_BN,                 // Parameters
            1);                                 // Import index

    backprop_conv2d(224, 112, 3, 3, 32,         // Line size, column size, kernel size, input depth, output depth
            image,                              // Input
            var.initial_conv2d,                 // Output
            par.initial_par_conv2d,             // Weights
            2, 0, 1);                           // Stride, padding and import index

    printf("\n");
}

void test(int c, float predictions[c], float fc_w[1280][c], float fc_b[c], int n) {

    int NC = 0;                                                     // Number of correct predictions
    int (*conf)[c] = calloc(c, sizeof *conf);                       // Confusion matrix: actual (lines) x predicted (columns)
    float step = 0.01;                                              // Threshold Step
    int t_number = 1 / step;                                      // Number of points    (0.5 -> 1)
    int (*conf_arr)[2][t_number+1] = calloc(2, sizeof *conf_arr);   // Confusion matrix array
    loss = 0;                                                       // Reset loss

    // Test
    for (int i = 0; i < n; ++i) {
        rewind(parbin);
        printf("\nEpoch %d Test/Validation Image %d\n", epoch_count+1, i+1);
        import_image(imagetestbin);
        inference(class, predictions, fc_w, fc_b);

        // Add to Correct Predictions
        if (inf_idx == label) NC++;

        // Add to Confusion Matrix
        conf[label][inf_idx]++;

        // Add to Confusion Matrix Array on Thresholds
        if (c == 2) {
            for (int t_count = 0; t_count < t_number + 1; ++t_count) {
                float th = t_count * step;  // Current threshold
                int th_idx;                 // Index within threshold

                // Calculate th_idx
                if (inf_yes >= th) th_idx = 1;
                else th_idx = 0;

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
        ROC[1][0] = 1;
        ROC[0][0] = 1;
        float area = 0;                                         // Area Under Curve
        for (int t_count = 1; t_count < t_number + 1; ++t_count) {
            ROC[1][t_count] = (float)conf_arr[1][1][t_count] / (conf_arr[1][0][t_count] + conf_arr[1][1][t_count]); // TPR = TP / (FN + TP)     aka Sensitivity
            ROC[0][t_count] = (float)conf_arr[0][1][t_count] / (conf_arr[0][0][t_count] + conf_arr[0][1][t_count]); // FPR = FP / (TN + FP)     aka 1 - Specificity
            area += ( (ROC[1][t_count] + ROC[1][t_count-1]) / 2) * (ROC[0][t_count-1] - ROC[0][t_count]);           // Area Under Curve (trapezoid approximation) (negative if line goes to the right)
        }
        /*
        // Print confusion array matrixes
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
        char name[23]; // maximum number of characters is "../data/roc_dataxx.txt" = 22
        sprintf(name,"../data/roc_data%d.dat",epoch_count);
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
    loss = loss / n_val;
    printf("\nCross-entropy Loss = %f\n", loss);
    loss_plot(epoch_count);

}

void transfer() {

    int i = 1;    // Index for comparison

    int offset = (1000-class)*1281*FSIZE;    // (1000-class) + (1280*1000 - class*1280) worth of bytes      might be a plus one here based on testing
    fseek(parbin,-offset,SEEK_END);          // Send pointer to effective end of parameter file

    // Fully Connected
    fillRandom("fc_b", class*FSIZE);
    fillRandom("fc_w", class*1280*FSIZE);

    // Final Block
    if (frz >= ++i) {
        fillRandom("param52",   sizeof(par.final_par_conv2d_BN)/4);
        fillRandom("weights35", sizeof(par.final_par_conv2d)/4);
    }

    // Block 18
    if (frz >= ++i) {
        fillRandom("param51",    sizeof(par.block18_par_project_BN));
        fillRandom("weights34",  sizeof(par.block18_par_project));
        fillRandom("param50",    sizeof(par.block18_par_depth_BN));
        fillRandom("dweights17", sizeof(par.block18_par_depth));
        fillRandom("param49",    sizeof(par.block18_par_expand_BN));
        fillRandom("weights33",  sizeof(par.block18_par_expand));
    }

    // Block 17
    if (frz >= ++i) {
        fillRandom("param48",    sizeof(par.block17_par_project_BN));
        fillRandom("weights32",  sizeof(par.block17_par_project));
        fillRandom("param47",    sizeof(par.block17_par_depth_BN));
        fillRandom("dweights16", sizeof(par.block17_par_depth));
        fillRandom("param46",    sizeof(par.block17_par_expand_BN));
        fillRandom("weights31",  sizeof(par.block17_par_expand));
    }

    // Block 16
    if (frz >= ++i) {
        fillRandom("param45",    sizeof(par.block16_par_project_BN));
        fillRandom("weights30",  sizeof(par.block16_par_project));
        fillRandom("param44",    sizeof(par.block16_par_depth_BN));
        fillRandom("dweights15", sizeof(par.block16_par_depth));
        fillRandom("param43",    sizeof(par.block16_par_expand_BN));
        fillRandom("weights29",  sizeof(par.block16_par_expand));
    }

    // Block 15
    if (frz >= ++i) {
        fillRandom("param42",    sizeof(par.block15_par_project_BN));
        fillRandom("weights28",  sizeof(par.block15_par_project));
        fillRandom("param41",    sizeof(par.block15_par_depth_BN));
        fillRandom("dweights14", sizeof(par.block15_par_depth));
        fillRandom("param40",    sizeof(par.block15_par_expand_BN));
        fillRandom("weights27",  sizeof(par.block15_par_expand));
    }

    // Block 14
    if (frz >= ++i) {
        fillRandom("param39",    sizeof(par.block14_par_project_BN));
        fillRandom("weights26",  sizeof(par.block14_par_project));
        fillRandom("param38",    sizeof(par.block14_par_depth_BN));
        fillRandom("dweights13", sizeof(par.block14_par_depth));
        fillRandom("param37",    sizeof(par.block14_par_expand_BN));
        fillRandom("weights25",  sizeof(par.block14_par_expand));
    }

    // Block 13
    if (frz >= ++i) {
        fillRandom("param36",    sizeof(par.block13_par_project_BN));
        fillRandom("weights24",  sizeof(par.block13_par_project));
        fillRandom("param35",    sizeof(par.block13_par_depth_BN));
        fillRandom("dweights12", sizeof(par.block13_par_depth));
        fillRandom("param34",    sizeof(par.block13_par_expand_BN));
        fillRandom("weights23",  sizeof(par.block13_par_expand));
    }

    // Block 12
    if (frz >= ++i) {
        fillRandom("param33",    sizeof(par.block12_par_project_BN));
        fillRandom("weights22",  sizeof(par.block12_par_project));
        fillRandom("param32",    sizeof(par.block12_par_depth_BN));
        fillRandom("dweights11", sizeof(par.block12_par_depth));
        fillRandom("param31",    sizeof(par.block12_par_expand_BN));
        fillRandom("weights21",  sizeof(par.block12_par_expand));
    }

    // Block 11
    if (frz >= ++i) {
        fillRandom("param30",    sizeof(par.block11_par_project_BN));
        fillRandom("weights20",  sizeof(par.block11_par_project));
        fillRandom("param29",    sizeof(par.block11_par_depth_BN));
        fillRandom("dweights10", sizeof(par.block11_par_depth));
        fillRandom("param28",    sizeof(par.block11_par_expand_BN));
        fillRandom("weights19",  sizeof(par.block11_par_expand));
    }

    // Block 10
    if (frz >= ++i) {
        fillRandom("param27",    sizeof(par.block10_par_project_BN));
        fillRandom("weights18",  sizeof(par.block10_par_project));
        fillRandom("param26",    sizeof(par.block10_par_depth_BN));
        fillRandom("dweights9",  sizeof(par.block10_par_depth));
        fillRandom("param25",    sizeof(par.block10_par_expand_BN));
        fillRandom("weights17",  sizeof(par.block10_par_expand));
    }

    // Block 9
    if (frz >= ++i) {
        fillRandom("param24",    sizeof(par.block9_par_project_BN));
        fillRandom("weights16",  sizeof(par.block9_par_project));
        fillRandom("param23",    sizeof(par.block9_par_depth_BN));
        fillRandom("dweights8",  sizeof(par.block9_par_depth));
        fillRandom("param22",    sizeof(par.block9_par_expand_BN));
        fillRandom("weights15",  sizeof(par.block9_par_expand));
    }

    // Block 8
    if (frz >= ++i) {
        fillRandom("param21",    sizeof(par.block8_par_project_BN));
        fillRandom("weights14",  sizeof(par.block8_par_project));
        fillRandom("param20",    sizeof(par.block8_par_depth_BN));
        fillRandom("dweights7",  sizeof(par.block8_par_depth));
        fillRandom("param19",    sizeof(par.block8_par_expand_BN));
        fillRandom("weights13",  sizeof(par.block8_par_expand));
    }

    // Block 7
    if (frz >= ++i) {
        fillRandom("param18",    sizeof(par.block7_par_project_BN));
        fillRandom("weights12",  sizeof(par.block7_par_project));
        fillRandom("param17",    sizeof(par.block7_par_depth_BN));
        fillRandom("dweights6",  sizeof(par.block7_par_depth));
        fillRandom("param16",    sizeof(par.block7_par_expand_BN));
        fillRandom("weights11",  sizeof(par.block7_par_expand));
    }

    // Block 6
    if (frz >= ++i) {
        fillRandom("param15",    sizeof(par.block6_par_project_BN));
        fillRandom("weights10",  sizeof(par.block6_par_project));
        fillRandom("param14",    sizeof(par.block6_par_depth_BN));
        fillRandom("dweights5",  sizeof(par.block6_par_depth));
        fillRandom("param13",    sizeof(par.block6_par_expand_BN));
        fillRandom("weights9",   sizeof(par.block6_par_expand));
    }

    // Block 5
    if (frz >= ++i) {
        fillRandom("param12",    sizeof(par.block5_par_project_BN));
        fillRandom("weights8",   sizeof(par.block5_par_project));
        fillRandom("param11",    sizeof(par.block5_par_depth_BN));
        fillRandom("dweights4",  sizeof(par.block5_par_depth));
        fillRandom("param10",    sizeof(par.block5_par_expand_BN));
        fillRandom("weights7",   sizeof(par.block5_par_expand));
    }

    // Block 4
    if (frz >= ++i) {
        fillRandom("param9",     sizeof(par.block4_par_project_BN));
        fillRandom("weights6",   sizeof(par.block4_par_project));
        fillRandom("param8",     sizeof(par.block4_par_depth_BN));
        fillRandom("dweights3",  sizeof(par.block4_par_depth));
        fillRandom("param7",     sizeof(par.block4_par_expand_BN));
        fillRandom("weights5",   sizeof(par.block4_par_expand));
    }

    // Block 3
    if (frz >= ++i) {
        fillRandom("param6",     sizeof(par.block3_par_project_BN));
        fillRandom("weights4",   sizeof(par.block3_par_project));
        fillRandom("param5",     sizeof(par.block3_par_depth_BN));
        fillRandom("dweights2",  sizeof(par.block3_par_depth));
        fillRandom("param4",     sizeof(par.block3_par_expand_BN));
        fillRandom("weights3",   sizeof(par.block3_par_expand));
    }

    // Expanded Block
    if (frz >= ++i) {
        fillRandom("param3",     sizeof(par.expanded_par_project_BN));
        fillRandom("weights2",   sizeof(par.expanded_par_project));
        fillRandom("param2",     sizeof(par.block3_par_depth_BN));
        fillRandom("dweights1",  sizeof(par.expanded_par_depth));

    // Initial Block
        fillRandom("param1",     sizeof(par.initial_par_BN));
        fillRandom("weights1",   sizeof(par.initial_par_conv2d));
    }
    printf("\n");

    clock_t begin = clock();

    // Train
    float predictions[class];
    float (*fc_w)[class] = calloc(1280, sizeof *fc_w);
    float fc_b[class];

    for (epoch_count = 0; epoch_count < n_epoch; ++epoch_count) {
        for (int i = 0; i < n_img; ++i) {
            printf("\nEpoch %d Image %d\n",epoch_count + 1, i + 1);
            import_image(imagebin);
            rewind(parbin);
            train(class, predictions, fc_w, fc_b);
        }
        if (n_val) test(class, predictions, fc_w, fc_b, n_val);
        rewind(imagebin);
        rewind(imagetestbin);
    }

    clock_t end = clock();
    double train_time = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\nTrain time with validation: %fs\n", train_time);

    free(fc_w);
}

