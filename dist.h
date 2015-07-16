#pragma once

unsigned sizeof_dist1d_workspace(int n);
void dist1d(float *dst,float* src,int n, int stride, void* workspace);

unsigned sizeof_dist2d_workspace(const int *n);
void dist2d(float *dst,float *src,const int *n,const int* stride,void *workspace);

unsigned sizeof_distnd_workspace(const int *n,int ndim);
void distnd(float *dst,float *src,const int *n,const int* strides,int ndim,void *workspace);
