#ifndef _MOD_CORREL_H_
#define _MOD_CORREL_H_

#include <fftw3.h>
#include <omp.h>
#include <stdlib.h>

using namespace std;

class plan {
  // Wrapper for FFTW plans
  public:
    fftw_plan p_forward, p_forward2, p_backward;
    double *in, *in2, *back;
    fftw_complex *out, *out2;
    int n, nc;
//     int col, col2;
    omp_lock_t lock;
    
    plan(double *data, double *data2, int ncut/*, int c, int c2*/) {
	omp_init_lock(&lock);
	omp_set_lock(&lock);
	nc	= ncut + 1;
	n 	= ncut*2;
      
	in = (double*) fftw_malloc ( sizeof ( double ) * n );
	out = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * nc );
	in2 = (double*) fftw_malloc ( sizeof ( double ) * n );
	out2 = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * nc );
	back = (double*) fftw_malloc ( sizeof ( double ) * n );

	p_forward = fftw_plan_dft_r2c_1d ( n, in, out, FFTW_ESTIMATE );
	p_forward2 = fftw_plan_dft_r2c_1d ( n, in2, out2, FFTW_ESTIMATE );
	p_backward = fftw_plan_dft_c2r_1d ( n, out, back, FFTW_ESTIMATE );

	fftw_free(in);
	fftw_free(out);
	fftw_free(in2);
	fftw_free(out2);
	fftw_free(back);

	
	omp_unset_lock(&lock);
    }
    
    ~plan() {
	omp_destroy_lock(&lock);
	fftw_destroy_plan(p_forward);
	fftw_destroy_plan(p_forward2);
	fftw_destroy_plan(p_backward);
	fftw_cleanup();
    }
    
};

class Correl
{
  private:
	//	FastFourier * ftrans;

  public:
    Correl();
	~Correl();

  //! Perform correlation of observables A(t), and B(t):  C = <A(0)B(t)>
  /*!
    \param data1 Timeseries of observable A; given as a continuous array.
    \param data2 Timeseries of observable B; given as a continuous array.
    \param ncut Length of array data1 = length of array data2.
    \param ans "Answer"; array of length ncut containing the correlation.
    \param m { Smoothing method: 
    -# m<0  ... Apply no smoothing.
    -# m==0 ... Apply Savitzky-Golay filter in the fourier domain.
    -# m>0  ... Apply average over blocks of size m in the fourier domain.
    }
    \param ltc { Long tail correction: 
    -# ltc==0 ... Do nothing.
    -# ltc==1 ... Subtract the mean value of the timeseries A before the 
    correlation, ie. C = < (A(t)-<A>) x (B(t)-<B>) >
    -# ltc==2 ... Subtract mean of correlation function after correlation,
    ie. C = Co-<Co>
    }
   */
  void correl(double *data1, double *data2, int ncut, double *ans, int m, int ltc, int lowhigh, plan *p) const;
	//! Perform correlation of timeseries stored in d1 and d2 with length tmax. Avoid long tail correction.
	void correl(double d1[], double d2[], int tmax, double c[], int sm, int lhc, plan *p) const { correl(d1,d2,tmax,c,sm,0,lhc,p); }
	void correl(double data1[], double data2[], int ncut, double ans[], const int m, const int ltc, const int lowhigh) const;
  void correl_direct(double data1[], double data2[], int size, double corrData[]) const;
	void runningAverage(double[], int, int) const;
	void lowCut(double[], int) const;
	void highCut(double[], int, int) const;
	void savitzkyGolay(double[], int) const;
	void longTailCorrection(double[], double[], double[], int) const;
	double getAvgDoLongTailCorrection(double[], int) const;
	void longTailCorrection(double[], int) const;
	//	void correlate_angvelmf(double *coor, double *out, int natoms, int maxTime, double dt);
	void msd_angvelmf(double *coor, double *msd, int natoms, int maxTime, double dt);
	void xcorrel_angvelmf(double *coor1, double *coor2, double *xcf, int natoms1, int natoms2, int maxTime, double dt);
	void rel_angvelmf(double *coor1, double *coor2, double *wcorr_fun1, double *wcorr_fun2, double *wxcorr_fun, int natoms1, int natoms2, int maxTime, int limTime, double dt);
	void transform(double *data, int ncut, double *ans_re, double *ans_im, int ltc) const;
};
#endif
