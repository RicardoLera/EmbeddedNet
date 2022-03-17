#ifndef AC_H
#define AC_H

void inference(int c, float predictions[c], float fc_w[1280][c], float fc_b[c]);
void train(int c, float predictions[c], float fc_w[1280][c], float fc_b[c]);
void test(int c, float predictions[c], float fc_w[1280][c], float fc_b[c], int n);
void transfer();

#endif
