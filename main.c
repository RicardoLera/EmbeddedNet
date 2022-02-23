#include <stdio.h>
#include <string.h>
#include "actions.h"
#include "data_manip.h"
#include "var.h"

// Initialize epoch counter
int epoch_count = 0;

// Initialize freeze index
char frz[4];

// Initialize transfer indexes
char tfr[4];
char class[4];
char hidden[4];

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
            if ((   (strcmp(argv[2],"b18") == 0) ||
                    (strcmp(argv[2],"b17") == 0) ||
                    (strcmp(argv[2],"b16") == 0) ||
                    (strcmp(argv[2],"b15") == 0) ||
                    (strcmp(argv[2],"b14") == 0) ||
                    (strcmp(argv[2],"b13") == 0) ||
                    (strcmp(argv[2],"b12") == 0) ||
                    (strcmp(argv[2],"b11") == 0) ||
                    (strcmp(argv[2],"b10") == 0) ||
                    (strcmp(argv[2],"b9")  == 0) ||
                    (strcmp(argv[2],"b8")  == 0) ||
                    (strcmp(argv[2],"b7")  == 0) ||
                    (strcmp(argv[2],"b6")  == 0) ||
                    (strcmp(argv[2],"b5")  == 0) ||
                    (strcmp(argv[2],"b4")  == 0) ||
                    (strcmp(argv[2],"b3")  == 0) ||
                    (strcmp(argv[2],"exp") == 0) ))
            {
                strcpy(frz, argv[2]);
            }
            else
            {
                printf("Invalid index\n");
                return(0);
            }
        }
        else {
            strcpy(frz, "no");
        }

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

        if ((   (strcmp(argv[2],"b18") == 0) ||
                (strcmp(argv[2],"b17") == 0) ||
                (strcmp(argv[2],"b16") == 0) ||
                (strcmp(argv[2],"b15") == 0) ||
                (strcmp(argv[2],"b14") == 0) ||
                (strcmp(argv[2],"b13") == 0) ||
                (strcmp(argv[2],"b12") == 0) ||
                (strcmp(argv[2],"b11") == 0) ||
                (strcmp(argv[2],"b10") == 0) ||
                (strcmp(argv[2],"b9")  == 0) ||
                (strcmp(argv[2],"b8")  == 0) ||
                (strcmp(argv[2],"b7")  == 0) ||
                (strcmp(argv[2],"b6")  == 0) ||
                (strcmp(argv[2],"b5")  == 0) ||
                (strcmp(argv[2],"b4")  == 0) ||
                (strcmp(argv[2],"b3")  == 0) ||
                (strcmp(argv[2],"exp") == 0) ))
        {
            strcpy(frz, argv[2]);
        }
        else {
            printf("Invalid freeze index\n");
            return(0);
        }

        if (strlen(argv[3]) <= 3)
            strcpy(class, argv[3]);
        else {
            printf("Maximum class value is 999\n");
            return(0);
        }

        if (strlen(argv[4]) <= 3)
            strcpy(hidden, argv[4]);
        else {
            printf("Maximum hidden value is 999\n");
            return(0);
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
