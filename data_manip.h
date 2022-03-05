#ifndef DM_H
#define DM_H

void import_image(int l, int t);
void import_conv2d (int id, int size, int od, int idx, float kdata[od][size][size][id]);
void import_bn(int depth, int idx, float pdata[4][depth]);
void import_moving(int depth, int idx, float pdata[4][depth]);
void import_depth(int depth, int ksize, int idx, float kdata[ksize][ksize][depth]);
void import_fc(int isize, int osize, float weight[isize][osize][2], float bias[osize][2]);

void export_fc(int d, int c, float w[d][c][2], float b[c][2]);
void export_depth(char* name, int ks, int d, float data[ks][ks][d]);
void export_bn(char* name, int d, float data[d][4]);
void export_conv2d(char* name, int d1, int d2, int d3, int d4, float data[d1][d2][d3][d4]);

void importTransfer();
void exportTransfer();
void copyLabels(char* path);
void fillRandom(char* name, int n);

#endif
