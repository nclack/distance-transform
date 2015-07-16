/* Distance transform 
   See: 

    Distance Transforms of Sampled Functions
    Pedro F. Felzenszwalb, Daniel P. Huttenlocher
    THEORY OF COMPUTING, Volume 8 (2012), pp. 415-428
*/

#include <float.h>
#include <string.h>

#ifdef _MSC_VER 
#define export __declspec(dllexport)
#else
#define export
#endif

struct parabola {
    int i;       /* the center of the parabola */
    float start; /* v[k],z[k] in the paper; the beginning of the interval where this parabola is the lower envelope */
};

export unsigned sizeof_dist1d_workspace(int n) {
    return (n+1)*sizeof(struct parabola);
}

/* worst case requires n+1 parabolas */
/* stride is in units of elements, as opposed to bytes */
export void dist1d(float *dst,float* src,int n, int stride, void* workspace) {
    struct parabola* parabolas=(struct parabola*)workspace;
    int i,k=0; /* the number of parabolas in the lower envelope */
    parabolas[0].i=0;
    parabolas[0].start=-FLT_MAX;
    parabolas[1].start=FLT_MAX;
    for(i=1;i<n;++i) {
        const int v=parabolas[k].i;
        const float i2=(float)i*i,
                    v2=(float)v*v,
                    denom=2.0f*i-2.0f*v,
                    s=(src[i*stride]+i2-(src[v*stride]+v2))/denom; /* test intersection point for k'th parabola */
        if(s<=parabolas[k].start) {                  /* Intersection is too far back, back up to check intersection with prev parabola */
            --k; --i;            
            continue;
        } else {
            ++k;
            parabolas[k].i=i;
            parabolas[k].start=s;
            parabolas[k+1].start=FLT_MAX;
        }
    }

    /* form output */
    for(i=0,k=0;i<n;++i) {
        while(parabolas[k+1].start<i) ++k;
        const float d=(float)(i-parabolas[k].i);
        dst[i*stride]=d*d+src[parabolas[k].i*stride];
    }
}

export unsigned sizeof_dist2d_workspace(const int *n) {
    int m=n[0]>n[1]?n[0]:n[1];
    return sizeof_dist1d_workspace(m);
}

export void dist2d(float *dst,float *src,const int *n,const int* strides, void *workspace) {
    int i;
    for(i=0;i<n[0];++i) 
        dist1d(dst+i*strides[0],src+i*strides[0],n[1],strides[1],workspace);
    for(i=0;i<n[1];++i) 
        dist1d(dst+i*strides[1],dst+i*strides[1],n[0],strides[0],workspace);
}

export unsigned sizeof_distnd_workspace(const int *n,int ndim) {
    int mx=0;
    int i;
    for(i=0;i<ndim;++i)
        mx=mx>n[i]?mx:n[i];
    return sizeof_dist1d_workspace(mx)+ndim*sizeof(*n);
}

static int* distnd_prod_n(const int *n,int ndim,void *workspace) {
    int i;
    int *p=(int*)workspace;
    memcpy(p,n,ndim*sizeof(*n));
    for(i=1;i<ndim;++i)
        p[i]*=p[i-1];
    return p;
}

static void* distnd_1d_workspace(int ndim, const void* workspace) {
    return ((char*)workspace)+ndim*sizeof(int);
}

export void distnd(float *dst,float *src,const int *n,const int* strides, int ndim, void *_workspace) {
    int i,j,d;
    int *p=distnd_prod_n(n,ndim,_workspace);
    void *workspace=distnd_1d_workspace(ndim,_workspace);
    memcpy(dst,src,strides[ndim]*sizeof(*src));
    for(d=0;d<ndim;++d) {
        for(j=0;j<p[ndim-1]/p[d];++j) {       
            float *_dst=dst+j*strides[d+1];
            if(d<1)
                dist1d(_dst,_dst,n[d],strides[d],workspace);
            else
                for(i=0;i<p[d-1];++i)
                dist1d(_dst+i*strides[0],_dst+i*strides[0],n[d],strides[d],workspace);
        }
    }
}

/* NOTES

Parallelizing

For D>1, Can do several columns at once.
For computing along D=0, can try to batch rows but run into cache problems.
    Need a divide and conquour strategy
    Might be straightforward to just join parabola lists, need some proof

*/


