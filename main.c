#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "actions.h"
#include "data_manip.h"
#include "var.h"

// Initialize epoch counter
int epoch_count = 0;

// Initialize freeze index
int frz;

// Initialize transfer indexes
int tfr;
int class;
int hidden;

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

        // Fill input using data0[y][x][d] syntax (y are lines, x are columns)
        import_image(0);

        inference();

    }
    else if (strcmp(argv[1],"train") == 0 ) {

        if (argc > 3) {
            printf("Too many arguments for TRAIN command\n");
            return(0);
        }

        if (argc == 3) {
            if      ((strcmp(argv[2],"fc")  == 0)) {frz = 1;}
            else if ((strcmp(argv[2],"b18") == 0)) {frz = 2;}
            else if ((strcmp(argv[2],"b17") == 0)) {frz = 3;}
            else if ((strcmp(argv[2],"b16") == 0)) {frz = 4;}
            else if ((strcmp(argv[2],"b15") == 0)) {frz = 5;}
            else if ((strcmp(argv[2],"b14") == 0)) {frz = 6;}
            else if ((strcmp(argv[2],"b13") == 0)) {frz = 7;}
            else if ((strcmp(argv[2],"b12") == 0)) {frz = 8;}
            else if ((strcmp(argv[2],"b11") == 0)) {frz = 9;}
            else if ((strcmp(argv[2],"b10") == 0)) {frz = 10;}
            else if ((strcmp(argv[2],"b9")  == 0)) {frz = 11;}
            else if ((strcmp(argv[2],"b8")  == 0)) {frz = 12;}
            else if ((strcmp(argv[2],"b7")  == 0)) {frz = 13;}
            else if ((strcmp(argv[2],"b6")  == 0)) {frz = 14;}
            else if ((strcmp(argv[2],"b5")  == 0)) {frz = 15;}
            else if ((strcmp(argv[2],"b4")  == 0)) {frz = 16;}
            else if ((strcmp(argv[2],"b3")  == 0)) {frz = 17;}
            else if ((strcmp(argv[2],"exp") == 0)) {frz = 18;}
            else
            {
                printf("Invalid index\n");
                return(0);
            }
        }
        else
            frz = 0;

        importTransfer();

        import_image(0);

        train();

    }
    else if (strcmp(argv[1],"transfer") == 0 ) {

        if (argc != 5) {
            printf("TRANSFER command must have 3 extra arguments:\n"
                    " - tfr: Selects which layers to transfer the parameters into. Possible values are the same as frz.\n"
                    " - class: How many classification neurons are in the last layer of the new model.\n"
                    " - hidden: How many hidden neurons are in the hidden layer of the new model. 0 does not create a hidden layer.\n");
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
        if (class == 0 || class > 999 || class < 0) {
            printf("Invalid class index (1 - 999)\n");
            return(0);
        }

        hidden = strtol(argv[4], &temp, 10);
        if (hidden == 0 || hidden > 999 || hidden < 0) {
            if (strtol(argv[4], &temp, 10) != 0) {
                printf("Invalid hidden index (1 - 999)\n");
                return(0);
            }
        }

        exportTransfer();

        transfer();

    }
    else {
        printf("Invalid action\n");
        return 0;
    }
    return 0;
}
