#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "var.h"

// These functions depend on the existence of a unix-like file system

void import_image(FILE* file) {

    for (int y = 0; y < 224; ++y) // lines
    for (int x = 0; x < 224; ++x) // columns
    for (int d = 0; d < 3; ++d)   // depth
        fread(buf,FSIZE,1,file) ? image[y][x][d] = *(float*)&buf : (printf("Failed to read image\n"), exit(EXIT_FAILURE));

    for (int i = 0; i < LSIZE; ++i) buf[i] = fgetc(file);
    label = strtol((char*)buf, (char **) NULL, 10);
    printf("Label = %d\n", label);
}

void import_conv2d(int id, int ksize, int od, int idx, float kdata[od][ksize][ksize][id]) {
    for (int y = 0; y < ksize; ++y) // lines
    for (int x = 0; x < ksize; ++x) // columns
    for (int l = 0; l < id; ++l)    // input depth (layers)
    for (int d = 0; d < od; ++d)    // output depth (number of filters)
        fread(buf,FSIZE,1,parbin) ? kdata[d][y][x][l] = *(float*)&buf : (printf("Failed to read conv2d at idx = %d\n", idx), exit(EXIT_FAILURE));
}

void import_bn(int depth, int idx, float pdata[4][depth]) {
    // [0][:] is gamma, [1][:] is beta, [2][:] is moving mean, [3][:] is moving variance
    for (int y = 0; y < 4; ++y)     // type
    for (int x = 0; x < depth; ++x) // depth
        fread(buf,FSIZE,1,parbin) ? pdata[y][x] = *(float*)&buf : (printf("Failed to read batch normalization at idx = %d\n", idx), exit(EXIT_FAILURE));
}

void import_depth(int depth, int ksize, int idx, float kdata[ksize][ksize][depth]) {
    for (int y = 0; y < ksize; ++y) // lines
    for (int x = 0; x < ksize; ++x) // columns
    for (int d = 0; d < depth; ++d) // depth
        fread(buf,FSIZE,1,parbin) ? kdata[y][x][d] = *(float*)&buf : (printf("Failed to read depthwise at idx = %d\n", idx), exit(EXIT_FAILURE));
}

void import_fc(int isize, int osize, float weight[isize][osize], float bias[osize]) {
    for (int y = 0; y < isize; ++y) // input layer
    for (int x = 0; x < osize; ++x) // classification layer
        fread(buf,FSIZE,1,parbin) ? weight[y][x] = *(float*)&buf : (printf("Failed to read fc weights\n"), exit(EXIT_FAILURE));

    for (int x = 0; x < osize; ++x) // classification layer
        fread(buf,FSIZE,1,parbin) ? bias[x] = *(float*)&buf : (printf("Failed to read fc biases\n"), exit(EXIT_FAILURE));
}

void export_fc(int d, int c, float w[d][c], float b[c]) {
    for (int i = c-1; i >= 0; --i) {
        memcpy(buf,&b[i],FSIZE);
        fseek(parbin,-FSIZE,SEEK_CUR);
        fwrite(buf,FSIZE,1,parbin);
        fseek(parbin,-FSIZE,SEEK_CUR);
    }
    printf("Saved fc biases\n");
    for (int y = d-1; y >= 0; --y) {
    for (int x = c-1; x >= 0; --x) {
        memcpy(buf,&w[y][x],FSIZE);
        fseek(parbin,-FSIZE,SEEK_CUR);
        fwrite(buf,FSIZE,1,parbin);
        fseek(parbin,-FSIZE,SEEK_CUR);
    }}
    printf("Saved fc weights\n");
}

void export_depth(char* name, int ksize, int depth, float kdata[ksize][ksize][depth]) {
    for (int y = ksize-1; y >= 0; --y) {
    for (int x = ksize-1; x >= 0; --x) {
    for (int d = depth-1; d >= 0; --d) {
        memcpy(buf,&kdata[y][x][d],FSIZE);
        fseek(parbin,-FSIZE,SEEK_CUR);
        fwrite(buf,FSIZE,1,parbin);
        fseek(parbin,-FSIZE,SEEK_CUR);
    }}}
    printf("Saved %s\n", name);
}

void export_bn(char* name, int depth, float data[4][depth]) {
    for (int y = 3; y >= 0; --y) {
    for (int x = depth-1; x >= 0; --x) {
        memcpy(buf,&data[y][x],FSIZE);
        fseek(parbin,-FSIZE,SEEK_CUR);
        fwrite(buf,FSIZE,1,parbin);
        fseek(parbin,-FSIZE,SEEK_CUR);
    }}
    printf("Saved %s\n", name);
}

void export_conv2d(char* name, int od, int ksize, int id, float kdata[od][ksize][ksize][id]) {
    for (int y = ksize-1; y >= 0; --y) {
    for (int x = ksize-1; x >= 0; --x) {
    for (int l = id-1; l >= 0; --l) {
    for (int d = od-1; d >= 0; --d) {
        memcpy(buf,&kdata[d][y][x][l],FSIZE);
        fseek(parbin,-FSIZE,SEEK_CUR);
        fwrite(buf,FSIZE,1,parbin);
        fseek(parbin,-FSIZE,SEEK_CUR);
    }}}}
    printf("Saved %s\n", name);
}

void importClass() {
    // Open file
    FILE *fptr;
    fptr = fopen("../data/labels.txt", "r");
    if (fptr == NULL) {
        perror("../data/labels.txt");
        exit(EXIT_FAILURE);
    }

    // Go to end of file and disregard random-ass extra linebreak
    fseek(fptr,-1,SEEK_END);

    // Go back until you find last line break
    char ch = 0;
    while (ch != '\n') {
        fseek(fptr,-2,SEEK_CUR);
        ch = fgetc(fptr);
    }

    // Get value in string and convert
    char s[5];
    char* p;
    int i = 0;
    ch = fgetc(fptr);   // Discard space
    ch = fgetc(fptr);   // Move to first number
    for (i = 0; ch != ':'; ++i) {
        s[i] = ch;
        ch = fgetc(fptr);
    }
    s[i] = '\0';
    class = strtol(s, &p, 10) + 1;

    printf("Class = %d\n", class);

    fclose(fptr);
}

void copyLabels(char* path) {
    FILE *source, *target;
    char ch;

    source = fopen(path, "r");
    if (source == NULL) {
        perror(path);
        exit(EXIT_FAILURE);
    }

    target = fopen("../data/labels.txt", "w");
    if (target == NULL) {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    while ((ch = fgetc(source)) != EOF)
       fputc(ch, target);

    fclose(source);
    fclose(target);

    printf("Labels copied successfully\n");
}

void fillRandom(char* name, int n) {
    for (int i = 0; i < n; ++i) {
        float f = (float)rand()/RAND_MAX;
        memcpy(buf,&f,FSIZE);
        fseek(parbin,-FSIZE,SEEK_CUR);
        fwrite(buf,FSIZE,1,parbin);
        fseek(parbin,-FSIZE,SEEK_CUR);
    }
    printf("Randomized %s\n", name);
}
