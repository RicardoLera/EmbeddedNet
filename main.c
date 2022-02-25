#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "actions.h"
#include "data_manip.h"
#include "var.h"

// Initialize epoch counter
int epoch_count;

// Initialize training indexes
int n_img = 1;
int n_epoch = 1;
int frz;

// Initialize transfer indexes
int tfr;
int class;

// Initialize image and label
float image[224][224][3];
int label;

int main( int argc, char *argv[] ) { 

    // Debugging
    // argc = 2;
    // argv[1] = "run";

    if (strcmp(argv[1],"run") == 0 ) {

        if ( argc > 2) {
            printf("Too many arguments for RUN command\n");
            return(0);
        }

        importTransfer();

        // Allocate FC
        float predictions[class];
        float (*fc_w)[class][2] = calloc(1280, sizeof *fc_w);
        float (*fc_b)[2] = calloc(class, sizeof *fc_b);

        // Fill input using data0[y][x][d] syntax (y are lines, x are columns)
        import_image(0);

        inference(class, predictions, fc_w, fc_b);

    }
    else if (strcmp(argv[1],"train") == 0 ) {

        if (argc > 5) {
            printf("Too many arguments for TRAIN command\n");
            return(0);
        }

        char *temp;

        if (argc >= 3) {
            if (strtol(argv[2], &temp, 10) < 1 || strtol(argv[2], &temp, 10) > 20) {
                printf("Invalid number of images (1 - 20)\n");
                return(0);
            }
            else
                n_img = strtol(argv[2], &temp, 10);
        }
        printf("%d\n", n_img);

        if (argc >= 4) {
            if (strtol(argv[3], &temp, 10) < 1 || strtol(argv[3], &temp, 10) > 50) {
                printf("Invalid number of epochs (1 - 50)\n");
                return(0);
            }
            else
                n_epoch = strtol(argv[3], &temp, 10);
        }

        if (argc == 5) {
            if      (strcmp(argv[4],"fc")  == 0) {frz = 1;}
            else if (strcmp(argv[4],"b18") == 0) {frz = 2;}
            else if (strcmp(argv[4],"b17") == 0) {frz = 3;}
            else if (strcmp(argv[4],"b16") == 0) {frz = 4;}
            else if (strcmp(argv[4],"b15") == 0) {frz = 5;}
            else if (strcmp(argv[4],"b14") == 0) {frz = 6;}
            else if (strcmp(argv[4],"b13") == 0) {frz = 7;}
            else if (strcmp(argv[4],"b12") == 0) {frz = 8;}
            else if (strcmp(argv[4],"b11") == 0) {frz = 9;}
            else if (strcmp(argv[4],"b10") == 0) {frz = 10;}
            else if (strcmp(argv[4],"b9")  == 0) {frz = 11;}
            else if (strcmp(argv[4],"b8")  == 0) {frz = 12;}
            else if (strcmp(argv[4],"b7")  == 0) {frz = 13;}
            else if (strcmp(argv[4],"b6")  == 0) {frz = 14;}
            else if (strcmp(argv[4],"b5")  == 0) {frz = 15;}
            else if (strcmp(argv[4],"b4")  == 0) {frz = 16;}
            else if (strcmp(argv[4],"b3")  == 0) {frz = 17;}
            else if (strcmp(argv[4],"exp") == 0) {frz = 18;}
            else
            {
                printf("Invalid freeze index\n");
                return(0);
            }
        }
        else
            frz = 0;

        importTransfer();

        // Allocate FC
        float predictions[class];
        float (*fc_w)[class][2] = calloc(1280, sizeof *fc_w);
        float (*fc_b)[2] = calloc(class, sizeof *fc_b);

        for (epoch_count = 0; epoch_count < n_epoch; ++epoch_count) {
            printf("Epoch %d\n", epoch_count + 1);
            for (int i = 0; i < n_img; ++i) {
                printf("Image %d\n", i + 1);
                import_image(i);
                train(class, predictions, fc_w, fc_b);
                printf("\n");
            }
        }

    }
    else if (strcmp(argv[1],"transfer") == 0 ) {

        if (argc != 5) {
            printf("TRANSFER command must have 3 extra arguments:\n"
                    " - tfr: Selects which layers to transfer the parameters into. Possible values are the same as frz.\n"
                    " - class: How many classification neurons are in the last layer of the new model.\n"
                    " - label: Destination of new classification labels file.\n");
            return(0);
        }

        if      ((strcmp(argv[2],"fc")   == 0)) {tfr = 1;}
        else if ((strcmp(argv[2],"b18")  == 0)) {tfr = 2;}
        else if ((strcmp(argv[2],"b17")  == 0)) {tfr = 3;}
        else if ((strcmp(argv[2],"b16")  == 0)) {tfr = 4;}
        else if ((strcmp(argv[2],"b15")  == 0)) {tfr = 5;}
        else if ((strcmp(argv[2],"b14")  == 0)) {tfr = 6;}
        else if ((strcmp(argv[2],"b13")  == 0)) {tfr = 7;}
        else if ((strcmp(argv[2],"b12")  == 0)) {tfr = 8;}
        else if ((strcmp(argv[2],"b11")  == 0)) {tfr = 9;}
        else if ((strcmp(argv[2],"b10")  == 0)) {tfr = 10;}
        else if ((strcmp(argv[2],"b9")   == 0)) {tfr = 11;}
        else if ((strcmp(argv[2],"b8")   == 0)) {tfr = 12;}
        else if ((strcmp(argv[2],"b7")   == 0)) {tfr = 13;}
        else if ((strcmp(argv[2],"b6")   == 0)) {tfr = 14;}
        else if ((strcmp(argv[2],"b5")   == 0)) {tfr = 15;}
        else if ((strcmp(argv[2],"b4")   == 0)) {tfr = 16;}
        else if ((strcmp(argv[2],"b3")   == 0)) {tfr = 17;}
        else if ((strcmp(argv[2],"exp")  == 0)) {tfr = 18;}
        else
        {
            printf("Invalid freeze index\n");
            return(0);
        }

        char *temp;

        class = strtol(argv[3], &temp, 10);
        if (class < 1 || class > 1000) {
            printf("Invalid class index (1 - 1000)\n");
            return(0);
        }

        copyLabels(argv[4]);

        exportTransfer();

        transfer();

    }
    else {
        printf("Invalid action\n");
        return 0;
    }
    return 0;
}
