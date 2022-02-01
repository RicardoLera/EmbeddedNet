#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "var.h"

//for testing
#include <assert.h>
//#define NDEBUG


// These functions depend on the architecture of the processor and interface used







// These functions are for testing on the PC

// We import everything as CSV files, so these next functions are for importing from that format

void import_image(int l) {

    // Define name from index
    char name[12]; // maximum number of characters is "imagexx.csv" = 11
    sprintf(name, "image%d.csv", l+1);

    // open file
    FILE *fptr;
    fptr = fopen(name, "r");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    char c = fgetc(fptr); // generic char
    char s[15];           // mantissa, maximum number of characters is "-x.xxxxxxxe-xx" = 14

    for (int y = 0; y < 224; ++y) {         // lines
        for (int x = 0; x < 224; ++x) {     // columns
            for (int d = 0; d < 3; ++d) {   // depth
                // write string
                int i;
                for (i = 0; c != '\n' && c != ' '; ++i) {
                    assert( 0 <= i && i <= 14 );
                    s[i] = c;
                    c = fgetc(fptr);
                }
                s[i] = '\0';
                float f = atof(s);      // convert to float
                image[y][x][d] = f;     // save on array
                c = fgetc(fptr);
            }
        }
    }
    fclose(fptr);

    sprintf(name, "label%d.csv", l+1);

    fptr = fopen(name, "r");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    c = fgetc(fptr);
    char* a;    // pointer for strtol
    int i;
    for (i = 0; c != '\n' && c != ' '; ++i) {
        s[i] = c;
        c = fgetc(fptr);
    }
    s[i] = '\0';
    label = strtol(s,&a,10);
    fclose(fptr);
}

void import_weights(int id, int size, int od, int idx, float kdata[od][size][size][id]) {

    // Define name from index
    char name[14]; // maximum number of characters is "weightsxx.csv" = 13
    sprintf(name, "weights%d.csv", idx);

    // open file
    FILE *fptr;
    fptr = fopen(name, "r");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    char c = fgetc(fptr); // generic char
    char s[15];           // mantissa, maximum number of characters is "-x.xxxxxxxe-xx" = 14

    for (int y = 0; y < size; ++y) {            // lines
        for (int x = 0; x < size; ++x) {        // columns
            for (int l = 0; l < id; ++l) {      // input depth (layers)
                for (int d = 0; d < od; ++d) {  // output depth (number of filters)
                    // write string
                    int i;
                    for (i = 0; c != '\n' && c != ' '; ++i) {
                        assert( 0 <= i && i <= 14 );
                        s[i] = c;
                        c = fgetc(fptr);
                    }
                    s[i] = '\0';
                    float f = atof(s);      // convert to float
                    kdata[d][y][x][l] = f;  // save on array
                    c = fgetc(fptr);
                }
            }
        }
    }
    fclose(fptr);
}

void import_bn(int depth, int idx, float pdata[4][depth]) {

    // [0][:] is gamma, [1][:] is beta, [2][:] is moving mean, [3][:] is moving variance
    // NOTE: only gamma and beta are imported here, the other two are imported in the next function

    // Define name from index
    char name[12]; // maximum number of characters is "paramxx.csv" = 11
    sprintf(name, "param%d.csv", idx);

    // open file
    FILE *fptr;
    fptr = fopen(name, "r");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    char c = fgetc(fptr); // generic char
    char s[15];           // mantissa, maximum number of characters is "-x.xxxxxxxe-xx" = 14

    for (int t = 0; t < 2; ++t) {           // type
        for (int d = 0; d < depth; ++d) {   // depth
            // write string
            int i;
            for (i = 0; c != '\n' && c != ' '; ++i) {
                assert( 0 <= i && i <= 14 );
                s[i] = c;
                c = fgetc(fptr);
            }
            s[i] = '\0';
            float f = atof(s);      // convert to float
            pdata[t][d] = f;        // save on array
            c = fgetc(fptr);
        }
    }
    fclose(fptr);
}

void import_moving(int depth, int idx, float pdata[4][depth]) {

    // [0][:] is gamma, [1][:] is beta, [2][:] is moving mean, [3][:] is moving variance

    // Define name from index
    char name[12]; // maximum number of characters is "paramxx.csv" = 11
    sprintf(name, "param%d.csv", idx);

    // open file
    FILE *fptr;
    fptr = fopen(name, "r");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    char c = fgetc(fptr); // generic char
    char s[15];           // mantissa, maximum number of characters is "-x.xxxxxxxe-xx" = 14

    // Skip gamma and beta
    for (int t = 0; t < 2; ++t) {           // type
        for (int d = 0; d < depth; ++d) {   // depth
            for (int i = 0; c != '\n' && c != ' '; ++i) {
                c = fgetc(fptr);
            }
            c = fgetc(fptr);
        }
    }

    for (int t = 2; t < 4; ++t) {           // type
        for (int d = 0; d < depth; ++d) {   // depth
            // write string
            int i;
            for (i = 0; c != '\n' && c != ' '; ++i) {
                assert( 0 <= i && i <= 14 );
                s[i] = c;
                c = fgetc(fptr);
            }
            s[i] = '\0';
            float f = atof(s);      // convert to float
            pdata[t][d] = f;        // save on array
            c = fgetc(fptr);
        }
    }
    fclose(fptr);
}


void import_dweights(int depth, int ksize, int idx, float kdata[ksize][ksize][depth]) {

    // Define name from index
    char name[15]; // maximum number of characters is "dweightsxx.csv" = 14
    sprintf(name, "dweights%d.csv", idx);

    // open file
    FILE *fptr;
    fptr = fopen(name, "r");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    char c = fgetc(fptr); // generic char
    char s[15];           // mantissa, maximum number of characters is "-x.xxxxxxxe-xx" = 14

    for (int y = 0; y < ksize; ++y) {           // lines
        for (int x = 0; x < ksize; ++x) {       // columns
            for (int d = 0; d < depth; ++d) {   // depth
                // write string
                int i;
                for (i = 0; c != '\n' && c != ' '; ++i) {
                    assert( 0 <= i && i <= 14 );
                    s[i] = c;
                    c = fgetc(fptr);
                }
                s[i] = '\0';
                float f = atof(s);      // convert to float
                kdata[y][x][d] = f;     // save on array
                c = fgetc(fptr);
            }
        }
    }
    fclose(fptr);
}

void import_fc(int isize, int osize, float weight[isize][osize], float *bias) {

    // open file
    FILE *fptr;
    fptr = fopen("fc_w.csv", "r");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    char c = fgetc(fptr); // generic char
    char s[15];           // mantissa, maximum number of characters is "-x.xxxxxxxe-xx" = 14

    for (int y = 0; y < isize; ++y) {     // 1280
        for (int x = 0; x < osize; ++x) { // 1000
            // write string
            int i;
            for (i = 0; c != '\n' && c != ' '; ++i) {
                assert( 0 <= i && i <= 14 );
                s[i] = c;
                c = fgetc(fptr);
            }
            s[i] = '\0';
            float f = atof(s);    // convert to float
            weight[y][x] = f;     // save on array
            c = fgetc(fptr);
        }
    }
    fclose(fptr);

    // open file
    fptr = fopen("fc_b.csv", "r");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    c = fgetc(fptr); // generic char

    for (int x = 0; x < osize; ++x) { // 1000
        // write string
        int i;
        for (i = 0; c != '\n' && c != ' '; ++i) {
            assert( 0 <= i && i <= 14 );
            s[i] = c;
            c = fgetc(fptr);
        }
        s[i] = '\0';
        float f = atof(s);  // convert to float
        bias[x] = f;        // save on array
        c = fgetc(fptr);
    }
    fclose(fptr);
}

void export(char* name, int d1, int d2, int d3, int d4, float data[d1][d2][d3][d4]) {
    FILE *fptr;
    fptr = fopen(name, "w");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d3; ++k) {
                for (int l = 0; l < d4; ++l) {
                    fprintf(fptr, "%.7e\n", data[i][j][k][l]);
                }
            }
        }
    }
    fclose(fptr);
    printf("Saved %s\n", name);
}

void export2(char* name, int d1, int d2, int d3, int d4, float data[d1][d2][d3][d4]) {
    FILE *fptr;
    fptr = fopen(name, "w");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < d3; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d4; ++k) {
                for (int l = 0; l < d1; ++l) {
                    fprintf(fptr, "%.7e ", data[l][i][j][k]);
                }
            }
        }
    }
    fclose(fptr);
    printf("Saved %s\n", name);
}

void import(char* name, int d1, int d2, int d3, int d4, float data[d1][d2][d3][d4]) {
    FILE *fptr;
    fptr = fopen(name, "r");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    char c = fgetc(fptr); // generic char
    char s[15];           // mantissa, maximum number of characters is "-x.xxxxxxxe-xx" = 14

    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d3; ++k) {
                for (int l = 0; l < d4; ++l) {
                    // write string
                    int i;
                    for (i = 0; c != '\n' && c != ' '; ++i) {
                        assert( 0 <= i && i <= 14 );
                        s[i] = c;
                        c = fgetc(fptr);
                    }
                    s[i] = '\0';
                    float f = atof(s);      // convert to float
                    data[i][j][k][l] = f;   // save on array
                    c = fgetc(fptr);
                }
            }
        }
    }
    fclose(fptr);
}

void import_1D(char* name, int d, float* data) {
    FILE *fptr;
    fptr = fopen(name, "r");
    if (fptr == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    char c = fgetc(fptr); // generic char
    char s[15];           // mantissa, maximum number of characters is "-x.xxxxxxxe-xx" = 14

    for (int j = 0; j < d; ++j) {
        // write string
        int i;
        for (i = 0; c != '\n' && c != ' '; ++i) {
            assert( 0 <= i && i <= 14 );
            s[i] = c;
            c = fgetc(fptr);
        }
        s[i] = '\0';
        float f = atof(s);      // convert to float
        data[j] = f;   // save on array
        c = fgetc(fptr);
    }
    fclose(fptr);
}
