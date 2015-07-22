/* Distance transform 
   See: 

    Distance Transforms of Sampled Functions
    Pedro F. Felzenszwalb, Daniel P. Huttenlocher
    THEORY OF COMPUTING, Volume 8 (2012), pp. 415-428
*/

#include <float.h>
#include <string.h>

#ifdef _MSC_VER 
#include <intrin.h>
#define export __declspec(dllexport)
#else
#include <x86intrin.h>
#define export
#endif

struct parabola {
    int i;       /* the center of the parabola */
    float start; /* v[k],z[k] in the paper; the beginning of the interval where this parabola is the lower envelope */
};

struct batched_workspace {
    __m256i *i;
    __m256  *start;
};

static unsigned dist1d_batched_sizeof_workspace(int n) {
    /* FIXME: pad for alignment corrections if neccessary so user doesn't have to use aligned malloc */
    return sizeof(struct batched_workspace)+(n+1)*(sizeof(__m256)+sizeof(__m256i));
}

#include <stdint.h>
/* n must be divisible by 32? for both i and start to be aligned */
static struct batched_workspace* dist1d_format_workspace(void *workspace,unsigned n) {
    const unsigned o=(n+1)*(sizeof(__m256)+sizeof(__m256i));
    struct batched_workspace* out=(struct batched_workspace*)( (uint8_t*)workspace+o ); /* place table-index at the end*/
    /* FIXME: pad for alignment corrections if neccessary so user doesn't have to use aligned malloc        
    */
    out->i=(__m256i*)workspace;
    out->start=(__m256*)((uint8_t*)workspace+(n+1)*(sizeof(__m256i)));
    return out;
}

/* uses avx2 to do 8 1d transforms at once.
   dst and src must be aligned to a 32-byte boundary.
   stride must be divisible by 8.

   FIXME: rename batched to something more descriptive like vec8.
*/
#include <stddef.h>
static dist1d_batched(float *dst, float *src, int n, int stride, void *workspace) {
    struct batched_workspace* ws=dist1d_format_workspace(workspace,n);
    const float * end = src+n*stride;
    const float inf=FLT_MAX,minusinf=-FLT_MAX;
    int i;
    __m256i k=_mm256_set_epi32(7,6,5,4,3,2,1,0); /* the offset for j'th parabola for the i'th peice of work is k[i]=8*j+i. */
    ws->i[0]=_mm256_setzero_si256();
    ws->start[0]=_mm256_set1_ps(-FLT_MAX);
    ws->start[1]=_mm256_set1_ps(FLT_MAX);
    for(i=0;i<n;++i){
        const float _i=(float)i;
        const __m256i v=_mm256_i32gather_epi32(ws->i,k,1);
        __m256  i2=_mm256_broadcast_ss(&_i),
                v2=_mm256_cvtepi32_ps(v),
                denom;
        i2=_mm256_mul_ps(i2,i2);
        v2=_mm256_mul_ps(v2,v2);
        {
            __m256i a=_mm256_sub_epi32(_mm256_set1_epi32(i),v);
            a=_mm256_mul_epi32(_mm256_set1_epi32(2),a);
            denom=_mm256_cvtepi32_ps(a);
        }
// HERE
        

    }
}

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
                    denom=2.0f*(i-v),
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


