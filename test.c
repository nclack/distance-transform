#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include "dist.h"

#pragma warning(disable:4244)

#define countof(e) (sizeof(e)/sizeof(*(e)))

static void writef32(const char* filename, float* data, int n) {
    FILE *fp=fopen(filename,"wb");
    fwrite(data,sizeof(*data),n,fp);
    fclose(fp);
}

static void test1d() {
    float *src,*dst;
    void *ws;
    const int N=100;
    src=(float*)malloc(N*sizeof(*src));
    dst=(float*)malloc(N*sizeof(*dst));
    {
        int i;
        for(i=0;i<N;++i)
            src[i]=100.0f;
        for(i=(N/2-0.1*N);i<(N/2+0.1*N);++i)
            src[i]=0.0f;

        for(i=(N*0.8-0.05*N);i<(N*0.8+0.05*N);++i)
            src[i]=5.0f;

        for(i=(N*0.55-0.05*N);i<(N*0.55+0.05*N);++i)
            src[i]=-5.0f;
    }
    ws=malloc(sizeof_dist1d_workspace(N));
    dist1d(dst,src,N,1,ws);
    free(ws);
    
    writef32("src1d.f32",src,N);
    writef32("dst1d.f32",dst,N);
}

static void test2d() {
    float *src,*dst;
    const int n[2]={320,240};
    const int strides[3]={1,n[0],n[0]*n[1]};
    src=(float*)malloc(strides[2]*sizeof(*src));
    dst=(float*)malloc(strides[2]*sizeof(*dst));
    
    {
        int x,y,i;
        float *c;
        struct box { int x0,x1,y0,y1; float v; } boxes[]={
           {0,n[0],0,n[1],0.0f},
           
           {n[0]/2-0.1*n[0],
            n[0]/2+0.1*n[0],
            n[1]/2-0.1*n[1],
            n[1]/2+0.1*n[1], 10000.0f},

           {n[0]/3-0.2*n[0],
            n[0]/3+0.2*n[0],
            n[1]/3-0.2*n[1],
            n[1]/3+0.2*n[1], 10000.0f}
        };
        for(i=0;i<countof(boxes);++i) {
            struct box *box=boxes+i;
            for(y=0,c=src;y<n[1];++y){
                for(x=0;x<n[0];++x,++c){
                    if(box->x0<=x && x<box->x1 && box->y0<=y && y<box->y1)
                        *c=box->v;
                }
            }
        }
    }

    {
        void *ws=malloc(sizeof_dist2d_workspace(n));
        dist2d(dst,src,n,strides,ws);
        free(ws);
    }

    writef32("src2d.f32",src,strides[2]);
    writef32("dst2d.f32",dst,strides[2]);
}

static void test3d() {
    float *src,*dst;
    const int n[]={320,240,100};
    const int strides[]={1,n[0],n[0]*n[1],n[0]*n[1]*n[2]};
    const int ndim=countof(n);
    src=(float*)malloc(strides[ndim]*sizeof(*src));
    dst=(float*)malloc(strides[ndim]*sizeof(*dst));

    {
        int x,y,z;
        float *c;
        struct{ int x0,x1,y0,y1,z0,z1; } box={
            n[0]/2-0.1*n[0],
            n[0]/2+0.1*n[0],
            n[1]/2-0.1*n[1],
            n[1]/2+0.1*n[1],
            n[2]/2-0.1*n[2],
            n[2]/2+0.1*n[2],
        };
        for(z=0,c=src;z<n[2];++z){
            for(y=0;y<n[1];++y){
                for(x=0;x<n[0];++x,++c){
                    if(box.x0<=x && x<box.x1 &&
                       box.y0<=y && y<box.y1 &&
                       box.z0<=z && z<box.z1
                       )
                        *c=0.0f;
                    else
                        *c=10000.0f;
                }
            }
        }
    }

    {
        void *ws=malloc(sizeof_distnd_workspace(n,ndim));
        distnd(dst,src,n,strides,ndim,ws);
        free(ws);
    }

    writef32("src3d.f32",src,strides[ndim]);
    writef32("dst3d.f32",dst,strides[ndim]);
}

int main(int argc,char*argv[]) {
    (void)argc;
    (void)argv;
    test1d();
    test2d();
    test3d();
    return 0;
}
