#ifndef DM_H
#define DM_H

void import_image(int l);
void import_weights (int id, int size, int od, int idx, float kdata[od][size][size][id]);
void import_bn(int depth, int idx, float pdata[4][depth]);
void import_moving(int depth, int idx, float pdata[4][depth]);
void import_dweights(int depth, int ksize, int idx, float kdata[ksize][ksize][depth]);
void import_fc(int isize, int osize, float weight[isize][osize], float *bias);
void importTransfer();
void export(char* name, int d1, int d2, int d3, int d4, float data[d1][d2][d3][d4]);
void exportConv(char* name, int d1, int d2, int d3, int d4, float data[d1][d2][d3][d4]);
void exportTransfer();
void fillRandom(char* name, int n);

//void import(char* name, int d1, int d2, int d3, int d4, float data[d1][d2][d3][d4]);
//void import_1D(char* name, int d, float* data);

#endif
