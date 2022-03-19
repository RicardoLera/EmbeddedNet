#ifndef DM_H
#define DM_H

void import_image(int l, int t);
void import_conv2d (int id, int ksize, int od, int idx, float kdata[od][ksize][ksize][id]);
void import_bn(int depth, int idx, float pdata[4][depth]);
void import_depth(int depth, int ksize, int idx, float kdata[ksize][ksize][depth]);
void import_fc(int isize, int osize, float weight[isize][osize], float bias[osize]);

void export_fc(int d, int c, float w[d][c], float b[c]);
void export_depth(char* name, int ksize, int depth, float kdata[ksize][ksize][depth]);
void export_bn(char* name, int depth, float data[4][depth]);
void export_conv2d(char* name, int od, int ksize, int id, float kdata[od][ksize][ksize][id]);

void importTransfer();
void exportTransfer();
void copyLabels(char* path);
void fillRandom(char* name, int n);

#endif
