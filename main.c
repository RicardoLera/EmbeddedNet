#include <stdio.h>
#include <string.h>
#include "actions.h"
#include "data_manip.h"
#include "var.h"

// Initialize epoch counter
int epoch_count = 0;

// Initialize freeze index
char frz[4];

// Initialize image and label
float image[224][224][3];
int label;

int main( int argc, char *argv[] ) { 

    // Debugging
    // argc = 2;
    // argv[1] = "run";

    if ( strcmp(argv[1],"run") == 0 ) {

        if ( argc > 2) {
            printf("Too many arguments for RUN command\n");
            return(0);
        }

        // Fill input using data0[y][x][d] syntax (y are lines, x are columns)
        import_image(0);

        inference();

    }
    else if ( strcmp(argv[1],"train") == 0 ) {

        if ( argc > 3) {
            printf("Too many arguments for TRAIN command\n");
            return(0);
        }

        if ( argc == 3) {
            strcpy(frz, argv[2]);
        }
        else {
            strcpy(frz, "no");
        }

        import_image(0);

        train();

    }
    else {
        printf("Invalid action\n");
        return 0;
    }
    return 0;
}
