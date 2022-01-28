#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "var.h"
#include "actions.h"
#include "data_manip.h"

// for testing
#include "layers.h"

// Initialize training boolean
int training;

// Initialize counters
int epoch_count = 0;
int batch_count = 0;

int main( int argc, char *argv[] ) { 

    //Debugging
    //argc = 2;
    //argv[1] = "run";

    if ( argc != 2) {
        printf("Number of arguments should be exactly 1\n");
        return 0;
    }

    if ( strcmp(argv[1],"run") == 0 ) {

        training = 0;

        // fill input using data0[y][x][d] syntax (y are lines, x are columns)
        import_image(0);

        inference(batch[0].image);

    }
    else if ( strcmp(argv[1],"train") == 0 ) {

        training = 1;

        train();
    }
    else {
        printf("Invalid action\n");
        return 0;
    }
    return 0;
}
