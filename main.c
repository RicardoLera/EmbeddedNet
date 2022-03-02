#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "actions.h"
#include "data_manip.h"
#include "var.h"

// Initialize structs
struct variables var;       // Every layer-correspondent variable
struct parameters par[2];   // Every parameter and its respective moving squared mean

// Initialize epoch counter
int epoch_count;

// Initialize training indexes
int n_img = 100;
int n_epoch = 3;
int frz = 1;

// Initialize transfer index
int class = 2;

// Initialize image and label
float image[224][224][3];
int label;

// Pointer for string operations
char *temp;

int main(int argc, char *argv[]) {

        // RUN Option
    if (strcmp(argv[1],"run") == 0 ) {

        int img_idx = 100;

        if (argc > 3) {
            printf("Too many arguments for RUN command.\nOnly extra argument is image index, default: 100\n");
            exit(EXIT_FAILURE);
        }

        if (argc > 2) {
            if (strtol(argv[2], &temp, 10) < 0 || strtol(argv[2], &temp, 10) > 100) {
                printf("Invalid image index (0 - 100)\n");
                exit(EXIT_FAILURE);
            }
            else
                img_idx = strtol(argv[2], &temp, 10);
        }

        importTransfer();

        // Allocate FC
        float predictions[class];
        float (*fc_w)[class][2] = calloc(1280, sizeof *fc_w);
        float (*fc_b)[2] = calloc(class, sizeof *fc_b);

        // Fill input using data0[y][x][d] syntax (y are lines, x are columns)
        import_image(img_idx);

        inference(class, predictions, fc_w, fc_b);

    }


        // TRAIN Option
    else if (strcmp(argv[1],"train") == 0 ) {

        if (argc % 2 != 0) {
            printf("Invalid arguments for TRAIN command.\n"
                    " - I: Number of images (1 - 100). Default: 100\n"
                    " - E: Number of epochs (1 - 100). Default: 3\n"
                    " - F: Which layers to freeze. Possible values are: 'fc', 'b18'-'b3' and 'exp'. Default 'fc'\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 2; i < argc; i = i + 2) {

            if (strcmp(argv[i],"I") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 1 || strtol(argv[i+1], &temp, 10) > 100) {
                    printf("Invalid image number (1 - 100)\n");
                    exit(EXIT_FAILURE);
                }
                else
                    n_img = strtol(argv[i+1], &temp, 10);
            }

            else if (strcmp(argv[i],"E") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 1 || strtol(argv[i+1], &temp, 10) > 100) {
                    printf("Invalid epoch number (1 - 100)\n");
                    exit(EXIT_FAILURE);
                }
                else
                    n_epoch = strtol(argv[i+1], &temp, 10);
            }

            else if (strcmp(argv[i],"F") == 0) {
                if      (strcmp(argv[i+1],"fc")  == 0) {frz = 1;}
                else if (strcmp(argv[i+1],"b18") == 0) {frz = 2;}
                else if (strcmp(argv[i+1],"b17") == 0) {frz = 3;}
                else if (strcmp(argv[i+1],"b16") == 0) {frz = 4;}
                else if (strcmp(argv[i+1],"b15") == 0) {frz = 5;}
                else if (strcmp(argv[i+1],"b14") == 0) {frz = 6;}
                else if (strcmp(argv[i+1],"b13") == 0) {frz = 7;}
                else if (strcmp(argv[i+1],"b12") == 0) {frz = 8;}
                else if (strcmp(argv[i+1],"b11") == 0) {frz = 9;}
                else if (strcmp(argv[i+1],"b10") == 0) {frz = 10;}
                else if (strcmp(argv[i+1],"b9")  == 0) {frz = 11;}
                else if (strcmp(argv[i+1],"b8")  == 0) {frz = 12;}
                else if (strcmp(argv[i+1],"b7")  == 0) {frz = 13;}
                else if (strcmp(argv[i+1],"b6")  == 0) {frz = 14;}
                else if (strcmp(argv[i+1],"b5")  == 0) {frz = 15;}
                else if (strcmp(argv[i+1],"b4")  == 0) {frz = 16;}
                else if (strcmp(argv[i+1],"b3")  == 0) {frz = 17;}
                else if (strcmp(argv[i+1],"exp") == 0) {frz = 18;}
                else
                {
                    printf("Invalid freeze index\n");
                    exit(EXIT_FAILURE);
                }
            }

            else {
                printf("Invalid argument '%s'\n"
                        " - I: Number of images (1 - 100). Default: 100\n"
                        " - E: Number of epochs (1 - 100). Default: 3\n"
                        " - F: Which layers to freeze. Possible values are: 'fc', 'b18'-'b3' and 'exp'. Default 'fc'\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        }

        importTransfer();

        // Allocate FC
        float predictions[class];
        float (*fc_w)[class][2] = calloc(1280, sizeof *fc_w);
        float (*fc_b)[2] = calloc(class, sizeof *fc_b);

        for (epoch_count = 0; epoch_count < n_epoch; ++epoch_count) {
            for (int i = 0; i < n_img; ++i) {
                printf("Epoch %d Image %d\n",epoch_count + 1, i + 1);
                import_image(i);
                train(class, predictions, fc_w, fc_b);
                printf("\n");
            }
        }
        free(fc_w);
        free(fc_b);
    }


        // TRANSFER Option
    else if (strcmp(argv[1],"transfer") == 0 ) {

        char L[] = "newlabels.txt";

        if (argc % 2 != 0) {
            printf("Invalid arguments for TRANSFER command.\n"
                    " - I: Number of images (1 - 100). Default: 100\n"
                    " - E: Number of epochs (1 - 100). Default: 3\n"
                    " - F: Which layers to freeze. Possible values are: 'fc', 'b18'-'b3', 'exp' and 'no'. Default 'fc'\n"
                    " - C: How many classification neurons are in the last layer of the new model. Default: 2\n"
                    " - L: Destination of new classification labels file. Default: 'newlabels.txt'\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 2; i < argc; i = i + 2) {

            if (strcmp(argv[i],"I") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 0 || strtol(argv[i+1], &temp, 10) > 100) {
                    printf("Invalid image number (0 - 100)\n");
                    exit(EXIT_FAILURE);
                }
                else
                    n_img = strtol(argv[i+1], &temp, 10);
            }

            else if (strcmp(argv[i],"E") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 1 || strtol(argv[i+1], &temp, 10) > 100) {
                    printf("Invalid epoch number (1 - 100)\n");
                    exit(EXIT_FAILURE);
                }
                else
                    n_epoch = strtol(argv[i+1], &temp, 10);
            }

            else if (strcmp(argv[i],"F") == 0) {
                if      (strcmp(argv[i+1],"fc")  == 0) {frz = 1;}
                else if (strcmp(argv[i+1],"b18") == 0) {frz = 2;}
                else if (strcmp(argv[i+1],"b17") == 0) {frz = 3;}
                else if (strcmp(argv[i+1],"b16") == 0) {frz = 4;}
                else if (strcmp(argv[i+1],"b15") == 0) {frz = 5;}
                else if (strcmp(argv[i+1],"b14") == 0) {frz = 6;}
                else if (strcmp(argv[i+1],"b13") == 0) {frz = 7;}
                else if (strcmp(argv[i+1],"b12") == 0) {frz = 8;}
                else if (strcmp(argv[i+1],"b11") == 0) {frz = 9;}
                else if (strcmp(argv[i+1],"b10") == 0) {frz = 10;}
                else if (strcmp(argv[i+1],"b9")  == 0) {frz = 11;}
                else if (strcmp(argv[i+1],"b8")  == 0) {frz = 12;}
                else if (strcmp(argv[i+1],"b7")  == 0) {frz = 13;}
                else if (strcmp(argv[i+1],"b6")  == 0) {frz = 14;}
                else if (strcmp(argv[i+1],"b5")  == 0) {frz = 15;}
                else if (strcmp(argv[i+1],"b4")  == 0) {frz = 16;}
                else if (strcmp(argv[i+1],"b3")  == 0) {frz = 17;}
                else if (strcmp(argv[i+1],"exp") == 0) {frz = 18;}
                else if (strcmp(argv[i+1],"no")  == 0) {frz = 19;}
                else
                {
                    printf("Invalid freeze index\n");
                    exit(EXIT_FAILURE);
                }
            }

            else if (strcmp(argv[i],"C") == 0) {
                if (strtol(argv[i+1], &temp, 10) < 2 || strtol(argv[i+1], &temp, 10) > 1000) {
                    printf("Invalid class number (2 - 1000)\n");
                    exit(EXIT_FAILURE);
                }
                else
                    class = strtol(argv[i+1], &temp, 10);
            }

            else if (strcmp(argv[i],"L") == 0) {
                strcpy(L, argv[i+1]);
            }

            else {
                printf("Invalid argument '%s'\n"
                        " - I: Number of images (1 - 100). Default: 100\n"
                        " - E: Number of epochs (1 - 100). Default: 3\n"
                        " - F: Which layers to freeze. Possible values are: 'fc', 'b18'-'b3' and 'exp'. Default 'fc'\n"
                        " - C: How many classification neurons are in the last layer of the new model. Default: 2\n"
                        " - L: Destination of new classification labels file. Default: 'newlabels.txt'\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        }

        copyLabels(L);

        srand(time(NULL));

        exportTransfer();

        transfer();

    }
    else {
        printf("Invalid action\n");
        exit(EXIT_FAILURE);
    }
    return 0;
}
