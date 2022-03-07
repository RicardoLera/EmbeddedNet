#ifndef AC_H
#define AC_H

void inference(int c, float predictions[c], float fc_w[1280][c][2], float fc_b[c][2]);
void train(int c, float predictions[c], float fc_w[1280][c][2], float fc_b[c][2]);
void test(int c, float predictions[c], float fc_w[1280][c][2], float fc_b[c][2], int n);
void transfer();

#endif
