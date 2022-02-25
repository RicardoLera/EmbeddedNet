#ifndef VAR_H
#define VAR_H

// Count index
extern int epoch_count;

// Training indexes
extern int n_img;
extern int n_epoch;
extern int frz;

// Transfer indexes
extern int tfr;     // Same as frz
extern int class;   // Maximum 999

// Declare image and label
extern float image[224][224][3];
extern int label;

// The following indexes refer to the OUTPUT of the respective layers. These are necessary for the backpropagation step.
struct variables {
    // Initial Block
    float initial_conv2d[112][112][32];  // Used in BN backpropagation
    //float initial_BN[112][112][32];    // We don't need the output of BN because the ReLU6 backprop doesn't depend on its inputs
    float initial_relu[112][112][32];    // Used for depthwise backpropagation, and so on

    // Expanded Block
    float expanded_depth[112][112][32];
    //float expanded_depth_BN[112][112][32];
    float expanded_relu[112][112][32];
    float expanded_project[112][112][16];
    float expanded_project_BN[112][112][16]; // We need this one, however, because it doesn't precede a ReLU6

    // Block 3 (Stride 2)
    float block3_expand[112][112][96];
    //float block3_expand_BN[112][112][96];
    float block3_expand_relu[112][112][96];
    float block3_depth[56][56][96];
    //float block3_depth_BN[56][56][96];
    float block3_depth_relu[56][56][96];
    float block3_project[56][56][24];
    float block3_project_BN[56][56][24];

    // Block 4 (Stride 1)
    float block4_expand[56][56][144];
    //float block4_expand_BN[56][56][144];
    float block4_expand_relu[56][56][144];
    float block4_depth[56][56][144];
    //float block4_depth_BN[56][56][144];
    float block4_depth_relu[56][56][144];
    float block4_project[56][56][24];
    //float block4_project_BN[56][56][24];   // Don't need this one because of the add layer
    float block4_add[56][56][24];

    // Block 5 (Stride 2)
    float block5_expand[56][56][144];
    //float block5_expand_BN[56][56][144];
    float block5_expand_relu[56][56][144];
    float block5_depth[28][28][144];
    //float block5_depth_BN[28][28][144];
    float block5_depth_relu[28][28][144];
    float block5_project[28][28][32];
    float block5_project_BN[28][28][32];

    // Block 6 (Stride 1)
    float block6_expand[28][28][192];
    //float block6_expand_BN[28][28][192];
    float block6_expand_relu[28][28][192];
    float block6_depth[28][28][192];
    //float block6_depth_BN[28][28][192];
    float block6_depth_relu[28][28][192];
    float block6_project[28][28][32];
    //float block6_project_BN[28][28][32];
    float block6_add[28][28][32];

    // Block 7 (Stride 1)
    float block7_expand[28][28][192];
    //float block7_expand_BN[28][28][192];
    float block7_expand_relu[28][28][192];
    float block7_depth[28][28][192];
    //float block7_depth_BN[28][28][192];
    float block7_depth_relu[28][28][192];
    float block7_project[28][28][32];
    //float block7_project_BN[28][28][32];
    float block7_add[28][28][32];

    // Block 8 (Stride 2)
    float block8_expand[28][28][192];
    //float block8_expand_BN[28][28][192];
    float block8_expand_relu[28][28][192];
    float block8_depth[14][14][192];
    //float block8_depth_BN[14][14][192];
    float block8_depth_relu[14][14][192];
    float block8_project[14][14][64];
    float block8_project_BN[14][14][64];

    // Block 9 (Stride 1)
    float block9_expand[14][14][384];
    //float block9_expand_BN[14][14][384];
    float block9_expand_relu[14][14][384];
    float block9_depth[14][14][384];
    //float block9_depth_BN[14][14][384];
    float block9_depth_relu[14][14][384];
    float block9_project[14][14][64];
    //float block9_project_BN[14][14][64];
    float block9_add[14][14][64];

    // Block 10 (Stride 1)
    float block10_expand[14][14][384];
    //float block10_expand_BN[14][14][384];
    float block10_expand_relu[14][14][384];
    float block10_depth[14][14][384];
    //float block10_depth_BN[14][14][384];
    float block10_depth_relu[14][14][384];
    float block10_project[14][14][64];
    //float block10_project_BN[14][14][64];
    float block10_add[14][14][64];

    // Block 11 (Stride 1)
    float block11_expand[14][14][384];
    //float block11_expand_BN[14][14][384];
    float block11_expand_relu[14][14][384];
    float block11_depth[14][14][384];
    //float block11_depth_BN[14][14][384];
    float block11_depth_relu[14][14][384];
    float block11_project[14][14][64];
    //float block11_project_BN[14][14][64];
    float block11_add[14][14][64];

    // Block 12 (Stride 1)
    float block12_expand[14][14][384];
    //float block12_expand_BN[14][14][384];
    float block12_expand_relu[14][14][384];
    float block12_depth[14][14][384];
    //float block12_depth_BN[14][14][384];
    float block12_depth_relu[14][14][384];
    float block12_project[14][14][96];
    //float block12_project_BN[14][14][96];
    float block12_add[14][14][96];           // Add block is deactivated in this one, but I keep the name

    // Block 13 (Stride 1)
    float block13_expand[14][14][576];
    //float block13_expand_BN[14][14][576];
    float block13_expand_relu[14][14][576];
    float block13_depth[14][14][576];
    //float block13_depth_BN[14][14][576];
    float block13_depth_relu[14][14][576];
    float block13_project[14][14][96];
    //float block13_project_BN[14][14][96];
    float block13_add[14][14][96];

    // Block 14 (Stride 1)
    float block14_expand[14][14][576];
    //float block14_expand_BN[14][14][576];
    float block14_expand_relu[14][14][576];
    float block14_depth[14][14][576];
    //float block14_depth_BN[14][14][576];
    float block14_depth_relu[14][14][576];
    float block14_project[14][14][96];
    //float block14_project_BN[14][14][96];
    float block14_add[14][14][96];

    // Block 15 (Stride 2)
    float block15_expand[14][14][576];
    //float block15_expand_BN[14][14][576];
    float block15_expand_relu[14][14][576];
    float block15_depth[7][7][576];
    //float block15_depth_BN[7][7][576];
    float block15_depth_relu[7][7][576];
    float block15_project[7][7][160];
    float block15_project_BN[7][7][160];

    // Block 16 (Stride 1)
    float block16_expand[7][7][960];
    //float block16_expand_BN[7][7][960];
    float block16_expand_relu[7][7][960];
    float block16_depth[7][7][960];
    //float block16_depth_BN[7][7][960];
    float block16_depth_relu[7][7][960];
    float block16_project[7][7][160];
    //float block16_project_BN[7][7][160];
    float block16_add[7][7][160];

    // Block 17 (Stride 1)
    float block17_expand[7][7][960];
    //float block17_expand_BN[7][7][960];
    float block17_expand_relu[7][7][960];
    float block17_depth[7][7][960];
    //float block17_depth_BN[7][7][960];
    float block17_depth_relu[7][7][960];
    float block17_project[7][7][160];
    //float block17_project_BN[7][7][160];
    float block17_add[7][7][160];

    // Block 18 (Stride 1)
    float block18_expand[7][7][960];
    //float block18_expand_BN[7][7][960];
    float block18_expand_relu[7][7][960];
    float block18_depth[7][7][960];
    //float block18_depth_BN[7][7][960];
    float block18_depth_relu[7][7][960];
    float block18_project[7][7][320];
    //float block18_project_BN[7][7][320];
    float block18_add[7][7][320];            // Same here

    // Final Block
    float final_conv2d[7][7][1280];
    //float final_conv2d_BN[7][7][1280];
    float final_conv2d_relu[7][7][1280];
    float final_pooling[1280];
    //float predictions[1000];                // Allocated Dynamically due to Transfer
};

struct variables var;   // Done this way so that I don't have to instantiate every variable again


// The following is the 3.4M parameter struct
struct parameters {

    // Initial Block
    float initial_par_conv2d[32][3][3][3];
    float initial_par_BN[4][32];

    // Expanded Block
    float expanded_par_depth[3][3][32];
    float expanded_par_depth_BN[4][32];
    float expanded_par_project[16][1][1][32];
    float expanded_par_project_BN[4][16];

    // Block 3 (Stride 2)
    float block3_par_expand[96][1][1][16];
    float block3_par_expand_BN[4][96];
    float block3_par_depth[3][3][96];
    float block3_par_depth_BN[4][96];
    float block3_par_project[24][1][1][96];
    float block3_par_project_BN[4][24];

    // Block 4 (Stride 1)
    float block4_par_expand[144][1][1][24];
    float block4_par_expand_BN[4][144];
    float block4_par_depth[3][3][144];
    float block4_par_depth_BN[4][144];
    float block4_par_project[24][1][1][144];
    float block4_par_project_BN[4][24];

    // Block 5 (Stride 2)
    float block5_par_expand[144][1][1][24];
    float block5_par_expand_BN[4][144];
    float block5_par_depth[3][3][144];
    float block5_par_depth_BN[4][144];
    float block5_par_project[32][1][1][144];
    float block5_par_project_BN[4][32];

    // Block 6 (Stride 1)
    float block6_par_expand[192][1][1][32];
    float block6_par_expand_BN[4][192];
    float block6_par_depth[3][3][192];
    float block6_par_depth_BN[4][192];
    float block6_par_project[32][1][1][192];
    float block6_par_project_BN[4][32];

    // Block 7 (Stride 1)
    float block7_par_expand[192][1][1][32];
    float block7_par_expand_BN[4][192];
    float block7_par_depth[3][3][192];
    float block7_par_depth_BN[4][192];
    float block7_par_project[32][1][1][192];
    float block7_par_project_BN[4][32];

    // Block 8 (Stride 2)
    float block8_par_expand[192][1][1][32];
    float block8_par_expand_BN[4][192];
    float block8_par_depth[3][3][192];
    float block8_par_depth_BN[4][192];
    float block8_par_project[64][1][1][192];
    float block8_par_project_BN[4][64];

    // Block 9 (Stride 1)
    float block9_par_expand[384][1][1][64];
    float block9_par_expand_BN[4][384];
    float block9_par_depth[3][3][384];
    float block9_par_depth_BN[4][384];
    float block9_par_project[64][1][1][384];
    float block9_par_project_BN[4][64];

    // Block 10 (Stride 1)
    float block10_par_expand[384][1][1][64];
    float block10_par_expand_BN[4][384];
    float block10_par_depth[3][3][384];
    float block10_par_depth_BN[4][384];
    float block10_par_project[64][1][1][384];
    float block10_par_project_BN[4][64];

    // Block 11 (Stride 1)
    float block11_par_expand[384][1][1][64];
    float block11_par_expand_BN[4][384];
    float block11_par_depth[3][3][384];
    float block11_par_depth_BN[4][384];
    float block11_par_project[64][1][1][384];
    float block11_par_project_BN[4][64];

    // Block 12 (Stride 1)
    float block12_par_expand[384][1][1][64];
    float block12_par_expand_BN[4][384];
    float block12_par_depth[3][3][384];
    float block12_par_depth_BN[4][384];
    float block12_par_project[96][1][1][384];
    float block12_par_project_BN[4][96];

    // Block 13 (Stride 1)
    float block13_par_expand[576][1][1][96];
    float block13_par_expand_BN[4][576];
    float block13_par_depth[3][3][576];
    float block13_par_depth_BN[4][576];
    float block13_par_project[96][1][1][576];
    float block13_par_project_BN[4][96];

    // Block 14 (Stride 1)
    float block14_par_expand[576][1][1][96];
    float block14_par_expand_BN[4][576];
    float block14_par_depth[3][3][576];
    float block14_par_depth_BN[4][576];
    float block14_par_project[96][1][1][576];
    float block14_par_project_BN[4][96];

    // Block 15 (Stride 2)
    float block15_par_expand[576][1][1][96];
    float block15_par_expand_BN[4][576];
    float block15_par_depth[3][3][576];
    float block15_par_depth_BN[4][576];
    float block15_par_project[160][1][1][576];
    float block15_par_project_BN[4][160];

    // Block 16 (Stride 1)
    float block16_par_expand[960][1][1][160];
    float block16_par_expand_BN[4][960];
    float block16_par_depth[3][3][960];
    float block16_par_depth_BN[4][960];
    float block16_par_project[160][1][1][960];
    float block16_par_project_BN[4][160];

    // Block 17 (Stride 1)
    float block17_par_expand[960][1][1][160];
    float block17_par_expand_BN[4][960];
    float block17_par_depth[3][3][960];
    float block17_par_depth_BN[4][960];
    float block17_par_project[160][1][1][960];
    float block17_par_project_BN[4][160];

    // Block 18 (Stride 1)
    float block18_par_expand[960][1][1][160];
    float block18_par_expand_BN[4][960];
    float block18_par_depth[3][3][960];
    float block18_par_depth_BN[4][960];
    float block18_par_project[320][1][1][960];
    float block18_par_project_BN[4][320];

    // Final Block
    float final_par_conv2d[1280][1][1][320];
    float final_par_conv2d_BN[4][1280];
    //float final_par_fc_w[1280][1000];
    //float final_par_fc_b[1000];                // Allocated Dynamically due to Transfer
};

struct parameters par[2];   // 0 represents trainable parameters; 1 represents moving squared means

#endif
