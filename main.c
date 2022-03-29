#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "actions.h"
#include "operations.h"
#include "data_manip.h"
#include "var.h"

#define TRAIN_MESSAGE printf("Invalid arguments for TRAIN command.\n" \
                            " -I: Number of images (1 - 3000). Default: 100\n" \
                            " -E: Number of epochs (1 - 100). Default: 3\n" \
                            " -V: Number of validation images (0 - 400). Default: 50. 0 does not validate every epoch\n" \
                            " -LR: Learning Rate (0 - 10). Default 0.045\n" \
                            " -LD: Learning Rate Decay (0 - 1). Default 0.98\n" \
                            " -F: Which layers to freeze. Possible values are: 'fc', 'b18'-'b3' and 'exp'. Default 'fc'\n");

#define TRANSFER_MESSAGE printf("Invalid arguments for TRANSFER command.\n" \
                            " -I: Number of images (1 - 3000). Default: 100\n" \
                            " -E: Number of epochs (1 - 100). Default: 3\n" \
                            " -V: Number of validation images (0 - 400). Default: 50. 0 does not validate every epoch\n" \
                            " -LR: Learning Rate (0 - 10). Default 0.045\n" \
                            " -LD: Learning Rate Decay (0 - 1). Default 0.98\n" \
                            " -F: Which layers to freeze. Possible values are: 'fc', 'b18'-'b3', 'exp' and 'no'. Default 'fc'\n" \
                            " -C: How many classification neurons are in the last layer of the new model. Default: 2\n" \
                            " -LF: Destination of new classification labels file. Default: '../newlabels.txt'\n");

// Initialize structs
struct variables var;       // Every layer-correspondent variable
struct parameters par;      // Every parameter and its respective moving squared mean

// Initialize file pointers and buffer
FILE* parbin;
FILE* imagebin;
FILE* imagetestbin;
unsigned char buf[FSIZE];

// Initialize epoch counter
int epoch_count;

// Initialize training parameters
int n_img = 100;
int n_epoch = 3;
int frz = 1;
float lr = 0.045;           // original: 0.045
float lr_decay = 0.98;      // original: 0.98

// Initialize transfer index
int class = 2;

// Initialize test parameters
int n_val = 50;
int inf_idx;
float inf_pred;
float inf_correct;
float inf_yes;
float loss;

// Initialize image and label
float image[224][224][3];
unsigned int label;

// Pointer for string operations
char *temp;

int main(int argc, char* argv[]) {

    // Open files
    parbin = fopen("../data/par.bin", "r+");
    if (parbin == NULL) {
        perror("../data/par.bin");
        exit(EXIT_FAILURE);
    }
    imagebin = fopen("../data/image.bin", "r+");
    if (parbin == NULL) {
        perror("../data/image.bin");
        exit(EXIT_FAILURE);
    }
    imagetestbin = fopen("../data/imagetest.bin", "r+");
    if (parbin == NULL) {
        perror("../data/imagetest.bin");
        exit(EXIT_FAILURE);
    }

        // RUN Option
    if (strcmp(argv[1],"run") == 0 ) {

        clock_t begin = clock();

        int img_idx = 0;

        if (argc > 3) {
            printf("Too many arguments for RUN command.\nOnly extra argument is image index: 0 - 400, default: 0\n");
            exit(EXIT_FAILURE);
        }

        if (argc > 2) {
            if (strtol(argv[2], &temp, 10) < 0 || strtol(argv[2], &temp, 10) > 400) {
                printf("Invalid image index (0 - 400)\n");
                exit(EXIT_FAILURE);
            }
            else img_idx = strtol(argv[2], &temp, 10);
        }

        importClass();

        // Allocate FC
        float predictions[class];
        float (*fc_w)[class] = calloc(1280, sizeof *fc_w);
        float fc_b[class];

        // Skip to test image
        fseek(imagetestbin, (FSIZE*150528 + 2)*img_idx, SEEK_SET); // 224*224*3 = 150528        CHANGE TO LABEL SIZE LATER

        // Fill input using data0[y][x][d] syntax (y are lines, x are columns)
        import_image(imagetestbin);

        inference(class, predictions, fc_w, fc_b);

        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("\nInference time: %fs\n", time_spent);
    }


        // TRAIN Option
    else if (strcmp(argv[1],"train") == 0 ) {

        if (argc % 2 != 0) {
            TRAIN_MESSAGE
            exit(EXIT_FAILURE);
        }

        for (int i = 2; i < argc; i = i + 2) {

            if (strcmp(argv[i],"-I") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 1 || strtol(argv[i+1], &temp, 10) > 3000) {
                    printf("Invalid image number (1 - 3000)\n");
                    exit(EXIT_FAILURE);
                }
                else n_img = strtol(argv[i+1], &temp, 10);
            }

            else if (strcmp(argv[i],"-E") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 1 || strtol(argv[i+1], &temp, 10) > 100) {
                    printf("Invalid epoch number (1 - 100)\n");
                    exit(EXIT_FAILURE);
                }
                else n_epoch = strtol(argv[i+1], &temp, 10);
            }

            else if (strcmp(argv[i],"-V") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 0 || strtol(argv[i+1], &temp, 10) > 400) {
                    printf("Invalid validation image number (0 - 400)\n");
                    exit(EXIT_FAILURE);
                }
                else n_val = strtol(argv[i+1], &temp, 10);
            }

            else if (strcmp(argv[i],"-LR") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 0 || strtol(argv[i+1], &temp, 10) > 10) {
                    printf("Invalid learning rate (0 - 10)\n");
                    exit(EXIT_FAILURE);
                }
                else lr = atof(argv[i+1]);
            }

            else if (strcmp(argv[i],"-LD") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 0 || strtol(argv[i+1], &temp, 10) > 1) {
                    printf("Invalid learning rate decay (0 - 1)\n");
                    exit(EXIT_FAILURE);
                }
                else lr_decay = atof(argv[i+1]);
            }

            else if (strcmp(argv[i],"-F") == 0) {
                if      (strcmp(argv[i+1],"fc")  == 0) frz = 1;
                else if (strcmp(argv[i+1],"b18") == 0) frz = 2;
                else if (strcmp(argv[i+1],"b17") == 0) frz = 3;
                else if (strcmp(argv[i+1],"b16") == 0) frz = 4;
                else if (strcmp(argv[i+1],"b15") == 0) frz = 5;
                else if (strcmp(argv[i+1],"b14") == 0) frz = 6;
                else if (strcmp(argv[i+1],"b13") == 0) frz = 7;
                else if (strcmp(argv[i+1],"b12") == 0) frz = 8;
                else if (strcmp(argv[i+1],"b11") == 0) frz = 9;
                else if (strcmp(argv[i+1],"b10") == 0) frz = 10;
                else if (strcmp(argv[i+1],"b9")  == 0) frz = 11;
                else if (strcmp(argv[i+1],"b8")  == 0) frz = 12;
                else if (strcmp(argv[i+1],"b7")  == 0) frz = 13;
                else if (strcmp(argv[i+1],"b6")  == 0) frz = 14;
                else if (strcmp(argv[i+1],"b5")  == 0) frz = 15;
                else if (strcmp(argv[i+1],"b4")  == 0) frz = 16;
                else if (strcmp(argv[i+1],"b3")  == 0) frz = 17;
                else if (strcmp(argv[i+1],"exp") == 0) frz = 18;
                else
                {
                    printf("Invalid freeze index\n");
                    exit(EXIT_FAILURE);
                }
            }

            else {
                TRAIN_MESSAGE
                exit(EXIT_FAILURE);
            }

        }

        importClass();

        // Allocate FC
        float predictions[class];
        float (*fc_w)[class] = calloc(1280, sizeof *fc_w);
        float fc_b[class];

        clock_t begin = clock();

        // Train
        for (epoch_count = 0; epoch_count < n_epoch; ++epoch_count) {
            for (int i = 0; i < n_img; ++i) {
                printf("Epoch %d Image %d\n",epoch_count + 1, i + 1);
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


        // TRANSFER Option
    else if (strcmp(argv[1],"transfer") == 0 ) {

        char LF[] = "../newlabels.txt";

        if (argc % 2 != 0) {
            TRANSFER_MESSAGE
            exit(EXIT_FAILURE);
        }



        for (int i = 2; i < argc; i = i + 2) {

            if (strcmp(argv[i],"-I") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 0 || strtol(argv[i+1], &temp, 10) > 3000) {
                    printf("Invalid image number (0 - 3000)\n");
                    exit(EXIT_FAILURE);
                }
                else n_img = strtol(argv[i+1], &temp, 10);
            }

            else if (strcmp(argv[i],"-E") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 1 || strtol(argv[i+1], &temp, 10) > 100) {
                    printf("Invalid epoch number (1 - 100)\n");
                    exit(EXIT_FAILURE);
                }
                else n_epoch = strtol(argv[i+1], &temp, 10);
            }

            else if (strcmp(argv[i],"-V") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 0 || strtol(argv[i+1], &temp, 10) > 400) {
                    printf("Invalid validation image number (0 - 400)\n");
                    exit(EXIT_FAILURE);
                }
                else n_val = strtol(argv[i+1], &temp, 10);
            }

            else if (strcmp(argv[i],"-LR") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 0 || strtol(argv[i+1], &temp, 10) > 10) {
                    printf("Invalid learning rate (0 - 10)\n");
                    exit(EXIT_FAILURE);
                }
                else lr = atof(argv[i+1]);
            }

            else if (strcmp(argv[i],"-LD") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 0 || strtol(argv[i+1], &temp, 10) > 1) {
                    printf("Invalid learning rate decay (0 - 1)\n");
                    exit(EXIT_FAILURE);
                }
                else lr_decay = atof(argv[i+1]);
            }

            else if (strcmp(argv[i],"-F") == 0) {
                if      (strcmp(argv[i+1],"fc")  == 0) frz = 1;
                else if (strcmp(argv[i+1],"b18") == 0) frz = 2;
                else if (strcmp(argv[i+1],"b17") == 0) frz = 3;
                else if (strcmp(argv[i+1],"b16") == 0) frz = 4;
                else if (strcmp(argv[i+1],"b15") == 0) frz = 5;
                else if (strcmp(argv[i+1],"b14") == 0) frz = 6;
                else if (strcmp(argv[i+1],"b13") == 0) frz = 7;
                else if (strcmp(argv[i+1],"b12") == 0) frz = 8;
                else if (strcmp(argv[i+1],"b11") == 0) frz = 9;
                else if (strcmp(argv[i+1],"b10") == 0) frz = 10;
                else if (strcmp(argv[i+1],"b9")  == 0) frz = 11;
                else if (strcmp(argv[i+1],"b8")  == 0) frz = 12;
                else if (strcmp(argv[i+1],"b7")  == 0) frz = 13;
                else if (strcmp(argv[i+1],"b6")  == 0) frz = 14;
                else if (strcmp(argv[i+1],"b5")  == 0) frz = 15;
                else if (strcmp(argv[i+1],"b4")  == 0) frz = 16;
                else if (strcmp(argv[i+1],"b3")  == 0) frz = 17;
                else if (strcmp(argv[i+1],"exp") == 0) frz = 18;
                else if (strcmp(argv[i+1],"no")  == 0) frz = 19;
                else
                {
                    printf("Invalid freeze index\n");
                    exit(EXIT_FAILURE);
                }
            }

            else if (strcmp(argv[i],"-C") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 2 || strtol(argv[i+1], &temp, 10) > 1000) {
                    printf("Invalid class number (2 - 1000)\n");
                    exit(EXIT_FAILURE);
                }
                else class = strtol(argv[i+1], &temp, 10);
            }

            else if (strcmp(argv[i],"-LF") == 0)
                strcpy(LF, argv[i+1]);

            else {
                TRANSFER_MESSAGE
                exit(EXIT_FAILURE);
            }

        }

        copyLabels(LF);

        srand(1);               // Used for consistent results
        //srand(time(NULL));

        transfer();

    }

    // TEST Option
    else if (strcmp(argv[1],"test") == 0 ) {

        clock_t begin = clock();

        n_val = 400;    // Reusing validation number variable as test number

        if (argc > 3) {
            printf("Too many arguments for TEST command.\nOnly extra argument is image number: 1 - 400, default: 400\n");
            exit(EXIT_FAILURE);
        }

        if (argc > 2) {
            if (strtol(argv[2], &temp, 10) < 0 || strtol(argv[2], &temp, 10) > 399) {
                printf("Invalid image number (1 - 400)\n");
                exit(EXIT_FAILURE);
            }
            else n_val = strtol(argv[2], &temp, 10);
        }

        importClass();

        // Allocate FC
        float predictions[class];
        float (*fc_w)[class] = calloc(1280, sizeof *fc_w);
        float fc_b[class];

        test(class, predictions, fc_w, fc_b, n_val);

        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("\nTest time: %fs\n", time_spent);
    }
    else {
        printf("Invalid action. Valid actions are: run, train, transfer and test.\n");
        exit(EXIT_FAILURE);
    }

    fclose(parbin);
    return 0;
}
