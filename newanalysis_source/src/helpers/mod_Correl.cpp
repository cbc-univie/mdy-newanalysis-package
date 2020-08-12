// =========================================================
// This class provides correlation fuctions taken from 
// Numerical recipes
// 
// 										  Gregor Neumayr 2007
// ==========================================================

// memory leak debugging
#define N(a,b,c) new a;

#include "mod_Correl.h"
#include "BertholdHorn.h"
#include <iostream>
#include <assert.h>
#include <cstdlib>
#include <fstream>

// needed for memcpy
#include <cstring>
#include <cmath>

Correl::Correl()
{
	//	ftrans = N( FastFourier(), ftrans, "fastfourier");
}

Correl::~Correl()
{
	//	delete ftrans;
}

void Correl::correl(double *data1, double *data2, int ncut, double *ans, int m, int ltc, int lowhigh, plan *p) const
{
	int i;
	double *in, *in2;
	fftw_complex *out, *out2;
	double *back;/*, *back2;*/
// 	fftw_plan plan_forward, plan_forward2;
// 	fftw_plan plan_backward; //(only one array left after multiplication)
 
	int nc	= ncut + 1;
	int n 	= ncut*2;	
	
	if (ltc == 1)
	{
		longTailCorrection(data1, ncut);
		if (data1 != data2)
		{
			longTailCorrection(data2, ncut);
		}
	}

	in = (double*) fftw_malloc ( sizeof ( double ) * n );
	out = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * nc );
	in2 = (double*) fftw_malloc ( sizeof ( double ) * n );
	out2 = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * nc );
	back = (double*) fftw_malloc ( sizeof ( double ) * n );

	for (i=0; i < ncut; i++) in[i] = data1[i];
	for (i=ncut; i < n; i++) in[i] = 0.0;
	for (i=0; i < ncut; i++) in2[i] = data2[i];
	for (i=ncut; i < n; i++) in2[i] = 0.0;
	
	fftw_execute_dft_r2c ( p->p_forward, in, out );
	fftw_execute_dft_r2c ( p->p_forward2, in2, out2 );
	
	for ( i = 0; i < nc; i++ )
	{
		//complex multiplication:  (ft-series 1) x (conjugated ft-series 2)
		double a1 = out[i][0];
		double b1 = out[i][1];
		double a2 = out2[i][0];
		double b2 = -out2[i][1];
  		
		out[i][0] = a1*a2 - b1*b2;
		out[i][1] = a1*b2 + a2*b1;
	}

	if (m > 0)				runningAverage((double*)out,ncut,m);
	else if (m == 0)		savitzkyGolay((double*)out,ncut);

	if (lowhigh > 0)		lowCut((double*)out, lowhigh);
	else if (lowhigh < 0) 	highCut((double*)out, n, -lowhigh);
	
	fftw_execute_dft_c2r ( p->p_backward, out, back );

  //  1d DFT FFTW_FORWARD:
  // y_k = sum_j=0^(n-1) x_j * exp(-2*i*pi*j*k/n)
  //  1d DFT FFTW_BACKWARD:
  // x_j = sum_k=0^(n-1) y_k * exp(2*i*pi*j*k/n)
  // FFTW computes an unnormalized transform, in that there is no
  // coefficient in front of the summation in the DFT. In other words,
  // applying the forward and then the backward transform will
  // multiply the input by n. 
	for ( i = 0; i < ncut; i++ ) ans[i] = back[i] / ((double)(n) * (double) (ncut - i));

	fftw_free ( in );
	fftw_free ( in2 );
	fftw_free ( out );
	fftw_free ( out2 );
	fftw_free ( back );
	
	if (ltc == 2) longTailCorrection(data1, data2, ans, ncut);
	
	return;
}

void Correl::transform(double *data, int ncut, double *ans_re, double *ans_im, int ltc) const
{
	int i;
	double *in;
	fftw_complex *out;
	double *back;
	fftw_plan plan_forward;
 
	int nc	= ncut + 1;
	int n 	= ncut*2;	
	
	if (ltc == 1)
	{
		longTailCorrection(data, ncut);
	}

	in = (double*) fftw_malloc ( sizeof ( double ) * n );
	out = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * nc );

	plan_forward = fftw_plan_dft_r2c_1d ( n, in, out, FFTW_ESTIMATE );
	
	for (i=0; i < ncut; i++) in[i] = data[i];
	for (i=ncut; i < n; i++) in[i] = 0.0;
	
	fftw_execute_dft_r2c ( plan_forward, in, out );
	
	for ( i = 0; i < nc; i++ )
	{
	  ans_re[i] = out[i][0];
	  ans_im[i] = out[i][1];
	}

	//  1d DFT FFTW_FORWARD:
	// y_k = sum_j=0^(n-1) x_j * exp(-2*i*pi*j*k/n)
	//  1d DFT FFTW_BACKWARD:
	// x_j = sum_k=0^(n-1) y_k * exp(2*i*pi*j*k/n)
	// FFTW computes an unnormalized transform, in that there is no
	// coefficient in front of the summation in the DFT. In other words,
	// applying the forward and then the backward transform will
	// multiply the input by n. 
	//	for ( i = 0; i < ncut; i++ ) ans[i] = back[i] / ((double)(n) * (double) (ncut - i));

	fftw_free ( in );
	fftw_free ( out );
	
	return;
}

void Correl::highCut(double data[], int n, int np) const
{
	int i;
	for (i = n; i > np; i--)
	{
		data[i] = 0.0;
	}
	
}

void Correl::lowCut(double data[], int np) const
{
	int i;
	for(i = 0; i < np; i++)
	{
		data[i] = 0.0;
	}
}

void Correl::runningAverage(double data[], int n, int np) const
{
	int i;

	double *tmp = N( double[n], tmp, "runningAverage:tmp");
	
	for (i = 0; i < np*2; 	i+=2)
	{
		tmp[0]  +=  data[i];
		tmp[1]	+=	data[i+1];
	}

	for (i = 2; i < n-np;	i+=2)
	{
		tmp[i]  	=   tmp[i-2] - data[i-2] + data[i+np*2-2];
		tmp[i+1] 	= 	tmp[i-1] - data[i-1] + data[i+np*2-1];
	}
	
	for (i = 0; i < n-np; 		i++)    data[i] =   tmp[i]/(np*1.0);
	for (i=np; i >= 0; i--) data[n-i] = 0;

	delete [] tmp;
}

void Correl::longTailCorrection(double data[], int n) const
{
	double tmp = 0;
	int i;
	
	for (i = 0; i < n; i++)
	{
		tmp += data[i];
	}
	tmp = (tmp/n);
	
	for (i = 0; i < n; i++)
	{
		data[i] -= tmp;
	}
}

double Correl::getAvgDoLongTailCorrection(double data[], int n) const
{
	double avg = 0;
	int i;
	
	for (i = 0; i < n; i++)
	{
		avg += data[i];
	}
	avg = (avg/n);
	for (i = 0; i < n; i++)
	{
		data[i] -= avg;
	}
	return avg;
}

void Correl::longTailCorrection(double data1[], double data2[], double answer[] , int n) const
{
	double tmp1 = 0, tmp2 = 0;
	int i;
	
	for (i = 0; i < n; i++)
	{
		tmp1 += data1[i];
	}
	tmp1 = (tmp1/n);
	
	if (data1 != data2)
	{
		for (i = 0; i < n; i++)
		{
			tmp2 += data2[i];
		}
		tmp2 = (tmp2/n);
	}
	else
	{
		tmp2 = tmp1;
	}
	
	tmp1 = tmp1*tmp2;
	
	for (i = 0; i < n; i++)
	{
		answer[i] -= tmp1;
	}
}

void Correl::savitzkyGolay(double data[], int n) const
{
	int i,j;

	static double g[21] = 
	{
		0.35629588, 0.28966685, 0.22868436, 0.17334839, 0.12365895, 0.07961604,
		0.04121965, 0.00846979,-0.01863354,-0.04009034,-0.05590062,-0.06606437,
 	   -0.07058159,-0.06945229,-0.06267645,-0.05025409,-0.03218521,-0.00846979,
	   	0.02089215, 0.05590062, 0.09655562
	};
	
	double *tmp = N( double[n], tmp, "savitzkyGolay");
	

	for (i = 0; i < n-21; i+=2)
	{

		for(j = 0; j < 21; j++)
		{
			tmp[i] += g[j]*data[i+j*2];
			tmp[i+1] += g[j]*data[i+j*2+1];
		}
	}	
	
	memcpy(data,tmp,(n-21)*sizeof(double));
	for (i=20; i >= 0; i--) data[n-i] = 0;

	delete [] tmp;
}

void Correl::correl_direct(double data1[], double data2[], int size, double corrData[]) const
{
  // Allocation
//  double *corrData = new double[size]; // Has to be allocated before call to correl_direct

  // Initialisation
  for ( int i(0); i<size; ++i )
    corrData[i] = 0;

  // Direct correlation
  for ( int i1(0); i1<size; ++i1 )
  {
    for ( int i2(0); i2<=i1; ++i2 )
    {
      corrData[i1-i2] += data1[i1] * data2[i2];
    }
  }

  // Normalisation
  for ( int i(0); i<size; ++i )
  {
    corrData[i] /= (size-i);
  }
}

// void Correl::correlate_angvelmf(double *coor, double *out, int natoms, int maxTime, double dt)
// {
//   // maxTime == nstep !!!
    
//   // data_t* tsData=Timeseries.at(0)->getData();
//   // 	double* tsData2=Timeseries.at(1)->getData();

//   // 	unsigned int N1 = Timeseries.at(0)->getDimensionSize(1);
//   // 	unsigned int N2 = Timeseries.at(0)->getDimensionSize(2);
//   // 	unsigned int N3 = Timeseries.at(0)->getDimensionSize(3);
//   // 	unsigned int maxTime = Timeseries.at(0)->getMaxTime();
//   // 	double invDt=1/(2*Timeseries.at(0)->getTimeUnit());
//   // 	double dt=Timeseries.at(0)->getTimeUnit(); // testing

//   const int SmoothingMethod = -1;
//   const int ltc = 0;
//   const int LowHighCut = 0;

//   int N1 = 1, N2 = natoms;

//   double *tsData = new double[(maxTime-1)*9];
  
//   double invDt = 1.0/(2.0*dt);

//   for (int i1(0); i1 < maxTime; i1++) {
//     int cidx = i1*natoms*3;
//     double cent_geom[3] = { 0.0, 0.0, 0.0 };
//     for (int i2(0); i2 < natoms; i2++) {
//       int acidx = cidx + i2*3;
//       for (int i3(0); i3 < 3; i3++)
// 	cent_geom[i3] += coor[acidx+i3];
//     }
//     for (int i3(0); i3 < 3; i3++)
//       cent_geom[i3] /= double(natoms);
//     for (int i2(0); i2 < natoms; i2++) {
//       int acidx = cidx + i2*3;
//       for (int i3(0); i3 < 3; i3++)
// 	coor[acidx+i3] -= cent_geom[i3];
//     }
//     if (i1 != 0) { // first frame is template
//       GetRotation(&(tsData[(i1-1)*9]), natoms, coor, &coor[cidx], 0);
//     }
//   }

//   maxTime -= 1; // first frame is template, R is 0, start at second frame
  
//   double *corr = new double[(maxTime-2)*9]; // <- 9 instead of N3!
//   for (int t(0); t < (maxTime-2)*9; ++t){
//     corr[t] = 0;
//   }

//   double* wts = new double[(maxTime-2)*3]; // testing
//   double *wcorr = new double[(maxTime-2)]; // testing
//   for (int t(0); t < (maxTime-2)*3; ++t){
//     wts[t] = 0;
//   }
//   for (int t(0); t < (maxTime-2); ++t){
//     wcorr[t] = 0;
//   } // testing

// // 0: Rxx, 1: Rxy, 2: Rxz, 3: Ryx, 4: Ryy, 5: Ryz, 6: Rzx, 7: Rzy, 8: Rzz, 

//   double rt[9];
//   double w[9];
//   double* wmfts = new double[(maxTime-2)*3];

//   double* ts  = new double[maxTime-2];
//   double* ts2 = new double[maxTime-2];
//   double* acf = new double[maxTime-1];

//   for (int t=1; t<maxTime-1; ++t) {
//     unsigned int ta = (t-1)*9;
//     unsigned int tb = (t+1)*9;
//     unsigned int ti = t*9;
        
//     // ( dR(t)/dt )^T
//     for (int i=0; i<3; ++i) {
//       for (int j=0; j<3; ++j) {
// 	rt[3*j+i] = tsData[tb+3*i+j] - tsData[ta+3*i+j];
// 	rt[3*j+i] *= invDt;
//       }
//     }

//     // w = ( dR(t)/dt )^T * R(t)
//     for (int i=0; i<9; ++i) w[i] = 0.0;
//     for (int i=0; i<3; ++i) {
//       for (int j=0; j<3; ++j) {
// 	for (int k=0; k<3; ++k) {
// 	  w[3*i+j] += rt[3*i+k] * tsData[ti+3*k+j];
// 	}
//       }
//     }

//     // wx = -w23 = w32, wy = w13 = -w31, wz = -w12 = w21
//     double wm[3] = { (-w[5]+w[7])/2, (w[2]-w[6])/2, (-w[1]+w[3])/2 };
//     // wmf(t) = R(t)w(t); molecular frame angular velocity
//     double wmrot[3] = {0, 0, 0};
//     for (int i=0; i<3; ++i) {
//       for (int j=0; j<3; ++j) {
// 	wmrot[i] += tsData[ti+3*i+j] * wm[j];
//           }
//     }
//     for (int i=0; i<3; ++i) {
//       wmfts[(t-1)*3+i] = wmrot[i];
//       wts[(maxTime-2)*i+(t-1)] = wm[i]; // testing
//     }
//   }


//   for (int i=0; i<3; ++i) { // testing
//     correl(
// 	   wts+(maxTime-2)*i,
// 	   wts+(maxTime-2)*i,
// 	   (maxTime-2),acf,SmoothingMethod, ltc, LowHighCut);
//     for (int t=0; t<(maxTime-2); ++t) {
//       wcorr[t] += acf[t];
//     }
//   } // testing
  
//   // < wi(0)fi(0) x wj(t)fj(t) >
//   for (int i=0; i<3; ++i) {
//     for (int j=0; j<3; ++j) {
//       for (int k=0; k<3; ++k) {
	
// 	for (int t=0; t < (maxTime-2); t++){
// 	  unsigned int ta = (t+1)*9;
// 	  //j
// 	  ts[t] = wmfts[t*3+i]; //tsData[ta+3*i+0]*wmfts[t*3+0] + tsData[ta+3*i+1]*wmfts[t*3+1] + tsData[ta+3*i+2]*wmfts[t*3+2];
// 	  ts[t] *= tsData[ta+3*i+k];
// 	  //k
// 	  ts2[t] = wmfts[t*3+j]; //tsData[ta+3*j+0]*wmfts[t*3+0] + tsData[ta+3*j+1]*wmfts[t*3+1] + tsData[ta+3*j+2]*wmfts[t*3+2];
// 	  ts2[t] *= tsData[ta+3*j+k];
// 	}
// 	correl(ts,ts2,(maxTime-2),acf,SmoothingMethod, ltc, LowHighCut);
// 	for (int t=0; t < (maxTime-2); t++){
// 	  corr[t*9+3*i+j] += acf[t];
// 	}
	
//       }
//     }
//   } // < wi(0)fi(0) x wj(t)fj(t) >

//   double *corrData = new double[(maxTime-2)*18]; // <- 18 instead of N3!
//   for (int t(0); t < (maxTime-2)*18; ++t){
//     corrData[t] = 0;
//   }
//   //    for (int n1(0); n1 <  N1; ++n1) {
//   // for (int n2(0); n2 <  N2; ++n2) {
  
//   // < wi(0) x wj(t) >
//   for (int i(0); i < 3; ++i) {
//     for (int j(0); j < 3; ++j) {
//       for (int t(0); t < (maxTime-2); ++t) {
// 	//unsigned int ta = (t+1)*N1*N2*N3+n1*N2*N3+n2*N3;
// 	ts[t] = wmfts[t*3+i]; //tsData[ta+i*3+0]*wmfts[t*3+0] + tsData[ta+i*3+1]*wmfts[t*3+1] + tsData[ta+i*3+2]*wmfts[t*3+2];
// 	ts2[t]= wmfts[t*3+j]; //tsData[ta+j*3+0]*wmfts[t*3+0] + tsData[ta+j*3+1]*wmfts[t*3+1] + tsData[ta+j*3+2]*wmfts[t*3+2];
//       }
//       correl(ts,ts2,(maxTime-2),acf,SmoothingMethod, ltc, LowHighCut);
//       for (int t=0; t < (maxTime-2); ++t) {
// 	corrData[t*18+3*i+j] += acf[t];
//       }
      
//       for (int k(0); k < 3; ++k) {
// 	for (int t(0); t < (maxTime-2); ++t) {
// 	  unsigned int ta = (t+1)*9;
// 	  ts[t] = tsData[ta+i*3+k];
// 	  ts2[t] = tsData[ta+j*3+k];
// 	}
// 	correl(ts,ts2,(maxTime-2),acf,SmoothingMethod, ltc, LowHighCut);
// 	  for (int t=0; t < (maxTime-2); t++) {
// 	    corrData[t*18+9+3*i+j] += acf[t];
// 	  }
//       }
      
//     }
//   }
  

//   delete ts;
//   delete ts2;
//   delete acf;
//   delete wmfts;
//   delete[] tsData;

//   double invNormConst = 1/((double)N1*(double)N2);
//   for (int m=0; m < (maxTime-2)*9; m++)
//   {
//     corr[m] *= invNormConst;
//   }

//   for (int m=0; m < (maxTime-2)*18; m++)
//   {
//     corrData[m] *= invNormConst;
//   }

//   for (int m=0; m < (maxTime-2); m++) wcorr[m] *= invNormConst; // testing

//   // std::string suffix=getOutfilenameSuffix();
//   // std::string fn = outfilename + "." + suffix + ".p";
//   // std::ofstream of(fn.c_str());
//   // double timeUnit = Timeseries.at(0)->getTimeUnit();

//   // for (int t(0); t < (maxTime-2); t++)
//   // {
//   //   of <<t*timeUnit;
//   //   for (int i=0; i < 9; ++i)
//   //   {
//   //     of <<"\t" <<corr[t*9+i];
//   //   }
//   //   for (int i=0; i < 18; ++i)
//   //   {
//   //     of <<"\t" <<corrData[t*18+i];
//   //   }
//   //   of <<"\t" <<wcorr[t]; // testing
//   //   of <<"\n";
//   // }
//   // of.close();

//   delete corr;
//   delete corrData;
//   corrData=NULL;

//   delete wts; // testing
//   delete wcorr; // testing

// }

void Correl::correl(double data1[], double data2[], int ncut, double ans[], const int m, const int ltc, const int lowhigh) const
{
	int i;
	double *in, *in2;
	fftw_complex *out, *out2;
	double *back;//, *back2;
	fftw_plan plan_forward, plan_forward2;
	fftw_plan plan_backward; //(only one array left after multiplication)
 
	int nc	= ncut + 1;
	int n 	= ncut*2;	
	
	if (ltc == 1)
	{
		longTailCorrection(data1, ncut);
		if (data1 != data2)
		{
			longTailCorrection(data2, ncut);
		}
	}

	in = (double*) fftw_malloc ( sizeof ( double ) * n );
	out = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * nc );
	in2 = (double*) fftw_malloc ( sizeof ( double ) * n );
	out2 = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * nc );
	back = (double*) fftw_malloc ( sizeof ( double ) * n );

	plan_forward = fftw_plan_dft_r2c_1d ( n, in, out, FFTW_ESTIMATE );
	plan_forward2 = fftw_plan_dft_r2c_1d ( n, in2, out2, FFTW_ESTIMATE );
	plan_backward = fftw_plan_dft_c2r_1d ( n, out, back, FFTW_ESTIMATE );

	for (i=0; i < ncut; i++) in[i] = data1[i];
	for (i=ncut; i < n; i++) in[i] = 0.0;
	for (i=0; i < ncut; i++) in2[i] = data2[i];
	for (i=ncut; i < n; i++) in2[i] = 0.0;
	
	fftw_execute ( plan_forward );
	fftw_execute ( plan_forward2 );
	
	for ( i = 0; i < nc; i++ )
	{
		//complex multiplication:  (ft-series 1) x (conjugated ft-series 2)
		double a1 = out[i][0];
		double b1 = out[i][1];
		double a2 = out2[i][0];
		double b2 = -out2[i][1];
  		
		out[i][0] = a1*a2 - b1*b2;
		out[i][1] = a1*b2 + a2*b1;
	}

	if (m > 0)				runningAverage((double*)out,ncut,m);
	else if (m == 0)		savitzkyGolay((double*)out,ncut);

	if (lowhigh > 0)		lowCut((double*)out, lowhigh);
	else if (lowhigh < 0) 	highCut((double*)out, n, -lowhigh);
	
	fftw_execute ( plan_backward );

  //  1d DFT FFTW_FORWARD:
  // y_k = sum_j=0^(n-1) x_j * exp(-2*i*pi*j*k/n)
  //  1d DFT FFTW_BACKWARD:
  // x_j = sum_k=0^(n-1) y_k * exp(2*i*pi*j*k/n)
  // FFTW computes an unnormalized transform, in that there is no
  // coefficient in front of the summation in the DFT. In other words,
  // applying the forward and then the backward transform will
  // multiply the input by n. 
	for ( i = 0; i < ncut; i++ ) ans[i] = back[i] / ((double)(n) * (double) (ncut - i));

	fftw_destroy_plan ( plan_forward );
	fftw_destroy_plan ( plan_forward2 );
	fftw_destroy_plan ( plan_backward );

	fftw_free ( in );
	fftw_free ( in2 );
	fftw_free ( out );
	fftw_free ( out2 );
	fftw_free ( back );

	fftw_cleanup();
	
	if (ltc == 2) longTailCorrection(data1, data2, ans, ncut);
	
	return;
}

void subCentGeom(double *coor, int &natoms, int &maxTime)
{
  for (int i1(0); i1 < maxTime; i1++) {
    int cidx = i1*natoms*3;
    double cent_geom[3] = { 0.0, 0.0, 0.0 };
    for (int i2(0); i2 < natoms; i2++) {
      int acidx = cidx + i2*3;
      for (int i3(0); i3 < 3; i3++) {
  	cent_geom[i3] += coor[acidx+i3];
      }
    }
    for (int i3(0); i3 < 3; i3++) {
      cent_geom[i3] /= double(natoms);
    }
    
    for (int i2(0); i2 < natoms; i2++) {
      int acidx = cidx + i2*3;
      for (int i3(0); i3 < 3; i3++)
  	coor[acidx+i3] -= cent_geom[i3];
    }
  }
}

void calcWmfts(double *tsData, double *wmfts, int &tn, double &invDt)
{
  double rt[9];
  double w[9];

  for (int t=1; t<tn-1; ++t) {
    unsigned int ta = (t-1)*9;
    unsigned int tb = (t+1)*9;
    unsigned int ti = t*9;
    
    // ( dR(t)/dt )^T
    for (int i=0; i<3; ++i) {
      for (int j=0; j<3; ++j) {
	rt[3*j+i] = tsData[tb+3*i+j] - tsData[ta+3*i+j];
	rt[3*j+i] *= invDt;
      }
    }
    
    // w = ( dR(t)/dt )^T * R(t)
    for (int i=0; i<9; ++i) w[i] = 0.0;
    for (int i=0; i<3; ++i) {
      for (int j=0; j<3; ++j) {
	for (int k=0; k<3; ++k) {
	  w[3*i+j] += rt[3*i+k] * tsData[ti+3*k+j];
	}
      }
    }
    
    // wx = -w23 = w32, wy = w13 = -w31, wz = -w12 = w21
    double wm[3] = { (-w[5]+w[7])/2, (w[2]-w[6])/2, (-w[1]+w[3])/2 };
    // wmf(t) = R(t)w(t); molecular frame angular velocity
    double wmrot[3] = {0, 0, 0};
    for (int i=0; i<3; ++i) {
      for (int j=0; j<3; ++j) {
	wmrot[i] += tsData[ti+3*i+j] * wm[j];
    	}
    }
    for (int i=0; i<3; ++i) {
      wmfts[(t-1)*3+i] = wmrot[i];
    }
  }
}

void calcIntegral(double *integral, double *wmfts, double &dt, int &tn)
{
  for (int t=0; t<tn*3; t++) { integral[t]=0.0; }
  
  for (int t=1; t<tn-2; t++) {
    unsigned int ta = (t-1)*3;
    unsigned int ti = t*3;
    
    for (int i=0; i<3; i++) {
      integral[ti+i] = integral[ta+i] + (wmfts[ta+i]+wmfts[ti+i])*0.5*dt;
    }
  }
}

void Correl::msd_angvelmf(double *coor, double *msd, int natoms, int maxTime, double dt)
{
  // maxTime == nstep !!!
  
  int *ctr = new int[maxTime*3];
  for (int i=0; i<maxTime*3; i++) ctr[i]=0;
  
  double invDt = 1.0/(2.0*dt);

  subCentGeom(coor,natoms,maxTime);
  
  //#pragma omp parallel for
  for (int tm=0; tm<maxTime-1; tm++)
  {
    int tn = maxTime  - tm;
    double *tsData = new double[tn*9];

    for (int ti=0; ti<tn; ti++)
    {
      GetRotation(&(tsData[ti*9]), natoms, &(coor[tm*natoms*3]), &(coor[(ti+tm)*natoms*3]), 0);
    }

    // 0: Rxx, 1: Rxy, 2: Rxz, 3: Ryx, 4: Ryy, 5: Ryz, 6: Rzx, 7: Rzy, 8: Rzz, 

    double* wmfts = new double[(tn-2)*3];

    calcWmfts(tsData, wmfts, tn, invDt);
    
    double *integral = new double[tn*3];

    calcIntegral(integral, wmfts, dt, tn);
    
    for (int t2=1; t2<tn-2; t2++) {
      for (int i=0; i<3; i++) {
     	int idx = t2*3 + i;
	int idx0 = (t2-1)*3 + i;
     	double tmp = std::pow(integral[3+i] - integral[idx], 2);
	//#pragma omp atomic
     	msd[idx0] += tmp;
	//#pragma omp atomic
     	ctr[idx0]++;
      }
    }

    delete[] tsData;
    delete[] wmfts;
    delete[] integral;

  }

  for (int i=0; i<maxTime; i++) {
    for (int j=0; j<3; j++) {
      msd[i*3+j] /= (double) ctr[i*3+j];
    }
  }

  delete[] ctr;
  
}

void Correl::xcorrel_angvelmf(double *coor1, double *coor2, double *xcf, int natoms1, int natoms2, int maxTime, double dt)
{
  // maxTime == nstep !!!
    
  int *ctr = new int[maxTime*3];
  for (int i=0; i<maxTime*3; i++) ctr[i]=0;
  
  double invDt = 1.0/(2.0*dt);

  subCentGeom(coor1,natoms1,maxTime);
  subCentGeom(coor2,natoms2,maxTime);
  
  //#pragma omp parallel for
  for (int tm=0; tm<maxTime-1; tm++)
  {
    int tn = maxTime  - tm;
    double *tsData1 = new double[tn*9];
    double *tsData2 = new double[tn*9];

    for (int ti=0; ti<tn; ti++)
    {
      GetRotation(&(tsData1[ti*9]), natoms1, &(coor1[tm*natoms1*3]), &(coor1[(ti+tm)*natoms1*3]), 0);
      GetRotation(&(tsData2[ti*9]), natoms2, &(coor2[tm*natoms2*3]), &(coor2[(ti+tm)*natoms2*3]), 0);
    }
    
    // 0: Rxx, 1: Rxy, 2: Rxz, 3: Ryx, 4: Ryy, 5: Ryz, 6: Rzx, 7: Rzy, 8: Rzz, 

    double* wmfts1 = new double[(tn-2)*3];
    double* wmfts2 = new double[(tn-2)*3];

    calcWmfts(tsData1, wmfts1, tn, invDt);
    calcWmfts(tsData2, wmfts2, tn, invDt);
    
    double *integral1 = new double[tn*3];
    double *integral2 = new double[tn*3];

    calcIntegral(integral1, wmfts1, dt, tn);
    calcIntegral(integral2, wmfts2, dt, tn);
    
    for (int t2=1; t2<tn-2; t2++) {
      for (int i=0; i<3; i++) {
     	int idx = t2*3 + i;
	int idx0 = (t2-1)*3 + i;
     	double tmp = (integral1[3+i] - integral1[idx]) * (integral2[3+i] - integral2[idx]);
	//#pragma omp atomic
     	xcf[idx0] += tmp;
	//#pragma omp atomic
     	ctr[idx0]++;
      }
    }

    delete[] integral1;
    delete[] integral2;
    delete[] tsData1;
    delete[] tsData2;
    delete[] wmfts1;
    delete[] wmfts2;

  }

  for (int i=0; i<maxTime; i++) {
    for (int j=0; j<3; j++) {
      xcf[i*3+j] /= (double) ctr[i*3+j];
    }
  }
    
  delete[] ctr;
  
}

// void Correl::rel_angvelmf(double *coor1, double *coor2, double *msd, double *wcorr_fun1, double *wcorr_fun2, double *wxcorr_fun, int natoms1, int natoms2, int maxTime, double dt)
// {
//   // maxTime == nstep !!!
    
//   int *ctr = new int[maxTime*3];
//   for (int i=0; i<maxTime*3; i++) ctr[i]=0;
  
//   double invDt = 1.0/(2.0*dt);

//   subCentGeom(coor1,natoms1,maxTime);
//   subCentGeom(coor2,natoms2,maxTime);
  
//   //#pragma omp parallel for
//   for (int tm=0; tm<maxTime-1; tm++)
//   {
//     int tn = maxTime  - tm;
//     double *tsData1 = new double[tn*9];
//     double *tsData2 = new double[tn*9];

//     for (int ti=0; ti<tn; ti++)
//     {
//       GetRotation(&(tsData1[ti*9]), natoms1, &(coor1[tm*natoms1*3]), &(coor1[(ti+tm)*natoms1*3]), 0);
//       GetRotation(&(tsData2[ti*9]), natoms2, &(coor2[tm*natoms2*3]), &(coor2[(ti+tm)*natoms2*3]), 0);
//     }
    
//     // 0: Rxx, 1: Rxy, 2: Rxz, 3: Ryx, 4: Ryy, 5: Ryz, 6: Rzx, 7: Rzy, 8: Rzz, 

//     double* wmfts1 = new double[(tn-2)*3];
//     double* wmfts2 = new double[(tn-2)*3];

//     calcWmfts(tsData1, wmfts1, tn, invDt);
//     calcWmfts(tsData2, wmfts2, tn, invDt);

//     // calculate wmf correlation functions
//     //double* ans = new double[(tn-2)];
//     //double* ans2 = new double[(tn-2)];
//     //double* tmp1 = new double[(tn-2)];
//     //double* tmp2 = new double[(tn-2)];

//     if (tn > 2)
//     {
//       for (int ti=0; ti<tn-3; ti++)
//       {
// 	for (int i=0; i<3; i++)
// 	{
// 	  wcorr_fun1[ti*3+i] += wmfts1[i] * wmfts1[ti*3+i];
// 	  wcorr_fun2[ti*3+i] += wmfts2[i] * wmfts2[ti*3+i];
// 	  wxcorr_fun[ti*3+i] += (wmfts1[i] * wmfts2[ti*3+i] + wmfts1[ti*3+i] * wmfts2[i]) * 0.5;
// 	}
//       }
//       // // wmfts1 self correlation
//       // // x
//       // for (int ti=0; ti<tn-2; ti++) tmp1[ti] = wmfts1[ti*3];
//       // correl(tmp1, tmp1, tn-2, ans, -1, 0, 0);
//       // for (int ti=0; ti<tn-3; ti++) wcorr_fun1[ti*3] += ans[ti];
//       // // y
//       // for (int ti=0; ti<tn-2; ti++) tmp1[ti] = wmfts1[ti*3+1];
//       // correl(tmp1, tmp1, tn-2, ans, -1, 0, 0);
//       // for (int ti=0; ti<tn-3; ti++) wcorr_fun1[ti*3+1] += ans[ti];
//       // // z
//       // for (int ti=0; ti<tn-2; ti++) tmp1[ti] = wmfts1[ti*3+2];
//       // correl(tmp1, tmp1, tn-2, ans, -1, 0, 0);
//       // for (int ti=0; ti<tn-3; ti++) wcorr_fun1[ti*3+2] += ans[ti];
      
//       // // wmfts2 self correlation
//       // // x
//       // for (int ti=0; ti<tn-2; ti++) tmp1[ti] = wmfts2[ti*3];
//       // correl(tmp1, tmp1, tn-2, ans, -1, 0, 0);
//       // for (int ti=0; ti<tn-3; ti++) wcorr_fun2[ti*3] += ans[ti];
//       // // y
//       // for (int ti=0; ti<tn-2; ti++) tmp1[ti] = wmfts2[ti*3+1];
//       // correl(tmp1, tmp1, tn-2, ans, -1, 0, 0);
//       // for (int ti=0; ti<tn-3; ti++) wcorr_fun2[ti*3+1] += ans[ti];
//       // // z
//       // for (int ti=0; ti<tn-2; ti++) tmp1[ti] = wmfts2[ti*3+2];
//       // correl(tmp1, tmp1, tn-2, ans, -1, 0, 0);
//       // for (int ti=0; ti<tn-3; ti++) wcorr_fun2[ti*3+2] += ans[ti];
      
//       // // wmfts1/wmfts2 cross correlation
//       // // x
//       // for (int ti=0; ti<tn-2; ti++) { tmp1[ti] = wmfts1[ti*3]; tmp2[ti] = wmfts2[ti*3]; }
//       // correl(tmp1, tmp2, tn-2, ans, -1, 0, 0);
//       // correl(tmp2, tmp1, tn-2, ans2, -1, 0, 0);
//       // for (int ti=0; ti<tn-3; ti++) wxcorr_fun[ti*3] += (ans[ti] + ans2[ti]) * 0.5;
//       // // y
//       // for (int ti=0; ti<tn-2; ti++) { tmp1[ti] = wmfts1[ti*3+1]; tmp2[ti] = wmfts2[ti*3+1]; }
//       // correl(tmp1, tmp2, tn-2, ans, -1, 0, 0);
//       // correl(tmp2, tmp1, tn-2, ans2, -1, 0, 0);
//       // for (int ti=0; ti<tn-3; ti++) wxcorr_fun[ti*3+1] += (ans[ti] + ans2[ti]) * 0.5;
//       // // z
//       // for (int ti=0; ti<tn-2; ti++) { tmp1[ti] = wmfts1[ti*3+2]; tmp2[ti] = wmfts2[ti*3+2]; }
//       // correl(tmp1, tmp2, tn-2, ans, -1, 0, 0);
//       // correl(tmp2, tmp1, tn-2, ans2, -1, 0, 0);
//       // for (int ti=0; ti<tn-3; ti++) wxcorr_fun[ti*3+2] += (ans[ti] + ans2[ti]) * 0.5;
//     }
    
//     // calculate delta wmfts

//     for (int ti=0; ti<tn-2; ti++)
//     {
//       for (int i=0; i<3; i++)
//       {
// 	wmfts1[ti*3+i] = wmfts2[ti*3+i] - wmfts1[ti*3+i];
//       }
//     }
    
//     double *integral = new double[tn*3];

//     calcIntegral(integral, wmfts1, dt, tn);
    
//     for (int t2=1; t2<tn-2; t2++) {
//       for (int i=0; i<3; i++) {
//      	int idx = t2*3 + i;
// 	int idx0 = (t2-1)*3 + i;
// 	double tmp = std::pow(integral[3+i] - integral[idx], 2);
// 	//#pragma omp atomic
//      	msd[idx0] += tmp;
// 	//#pragma omp atomic
//      	ctr[idx0]++;
//       }
//     }

//     delete[] integral;
//     delete[] tsData1;
//     delete[] tsData2;
//     delete[] wmfts1;
//     delete[] wmfts2;
//     //delete[] ans;
//     //delete[] ans2;
//     //delete[] tmp1;
//     //delete[] tmp2;

//   }

//   for (int i=0; i<maxTime; i++) {
//     for (int j=0; j<3; j++) {
//       msd[i*3+j] /= (double) ctr[i*3+j];
//       wcorr_fun1[i*3+j] /= (double) ctr[i*3+j];
//       wcorr_fun2[i*3+j] /= (double) ctr[i*3+j];
//       wxcorr_fun[i*3+j] /= (double) ctr[i*3+j];
//     }
//   }
    
//   delete[] ctr;
  
// }

void Correl::rel_angvelmf(double *coor1, double *coor2, double *wcorr_fun1, double *wcorr_fun2, double *wxcorr_fun, int natoms1, int natoms2, int maxTime, int limTime, double dt)
{
  // maxTime == nstep !!!
    
  double invDt = 1.0/(2.0*dt);

  //subCentGeom(coor1,natoms1,maxTime);
  //subCentGeom(coor2,natoms2,maxTime);

  long maxCorr = long(maxTime-limTime)*long(limTime);

  //std::cout << maxCorr << std::endl;
  
  double *wcorr1_tmp = new double[maxCorr];
  double *wcorr2_tmp = new double[maxCorr];
  double *xcorr_tmp = new double[maxCorr];

  for (int ti=0; ti<maxCorr; ti++) {
    wcorr1_tmp[ti] = 0.0;
    wcorr2_tmp[ti] = 0.0;
    xcorr_tmp[ti] = 0.0;
  }

#pragma omp parallel for schedule(dynamic)
  for (int tm=0; tm<maxTime-limTime; tm++)
  {
    int tn = limTime;
    double *tsData1 = new double[tn*9];
    double *tsData2 = new double[tn*9];

    for (int ti=0; ti<tn; ti++)
    {
      GetRotation(&(tsData1[ti*9]), natoms1, &(coor1[tm*natoms1*3]), &(coor1[(ti+tm)*natoms1*3]), 0);
      GetRotation(&(tsData2[ti*9]), natoms2, &(coor2[tm*natoms2*3]), &(coor2[(ti+tm)*natoms2*3]), 0);
    }
    
    // 0: Rxx, 1: Rxy, 2: Rxz, 3: Ryx, 4: Ryy, 5: Ryz, 6: Rzx, 7: Rzy, 8: Rzz, 

    double* wmfts1 = new double[(tn-2)*3];
    double* wmfts2 = new double[(tn-2)*3];

    calcWmfts(tsData1, wmfts1, tn, invDt);
    calcWmfts(tsData2, wmfts2, tn, invDt);

    // std::ofstream of("wmfts1_x_gr1_fine.dat");
    // for (int ti=0; ti<tn-2; ti++) {
    //   of << ti*0.1 << '\t' << wmfts1[ti*3] << std::endl;
    // }
    // exit(1);

    for (int ti=0; ti<tn-2; ti++)
    {
      int tc = tm*limTime+ti;
      for (int i=0; i<3; i++)
      {
	wcorr1_tmp[tc] += wmfts1[i] * wmfts1[ti*3+i];
	wcorr2_tmp[tc] += wmfts2[i] * wmfts2[ti*3+i];
	xcorr_tmp[tc]  += (wmfts1[i] * wmfts2[ti*3+i] + wmfts1[ti*3+i] * wmfts2[i]) * 0.5;
      }
      wcorr1_tmp[tc] /= 3.0;
      wcorr2_tmp[tc] /= 3.0;
      xcorr_tmp[tc] /= 3.0;
    }
    
    delete[] tsData1;
    delete[] tsData2;
    delete[] wmfts1;
    delete[] wmfts2;
  }

  for (int i=0; i<limTime-1; i++) {
    int ctr=0;
    for (int j=0; j<(maxTime-limTime); j++) {
      //std::cout << i << '\t' << j << '\t' << maxTime << '\t' << limTime << '\t' << j*limTime+i << '\t' << maxCorr << std::endl;
      wcorr_fun1[i] += wcorr1_tmp[j*limTime+i];
      wcorr_fun2[i] += wcorr2_tmp[j*limTime+i];
      wxcorr_fun[i] += xcorr_tmp[j*limTime+i];
      ctr++;
    }
    wcorr_fun1[i] /= double(ctr);
    wcorr_fun2[i] /= double(ctr);
    wxcorr_fun[i] /= double(ctr);
  }

  delete[] wcorr1_tmp;
  delete[] wcorr2_tmp;
  delete[] xcorr_tmp;
  
}


// void Correl::rel_angvelmf(double *coor1, double *coor2, double *coor2_template, double *msd, int natoms1, int natoms2, int maxTime, double dt)
// {
//   // maxTime == nstep !!!
    
//   int *ctr = new int[maxTime*3];
//   for (int i=0; i<maxTime*3; i++) ctr[i]=0;
  
//   double invDt = 1.0/(2.0*dt);

//   // substract cog of second coor set of both sets
//   for (int i1(0); i1 < maxTime; i1++) {
//     int cidx = i1*natoms2*3;
//     double cent_geom[3] = { 0.0, 0.0, 0.0 };
//     for (int i2(0); i2 < natoms2; i2++) {
//       int acidx = cidx + i2*3;
//       for (int i3(0); i3 < 3; i3++) {
//   	cent_geom[i3] += coor2[acidx+i3];
//       }
//     }
//     for (int i3(0); i3 < 3; i3++) {
//       cent_geom[i3] /= double(natoms2);
//     }
    
//     for (int i2(0); i2 < natoms2; i2++) {
//       int acidx = cidx + i2*3;
//       for (int i3(0); i3 < 3; i3++)
//   	coor2[acidx+i3] -= cent_geom[i3];
//     }
//     for (int i2(0); i2 < natoms1; i2++) {
//       int acidx = i1*natoms1*3 + i2*3;
//       for (int i3(0); i3 < 3; i3++)
//   	coor1[acidx+i3] -= cent_geom[i3];
//     }
//   }
  
//   //#pragma omp parallel for
//   for (int tm=0; tm<maxTime-1; tm++)
//   {
//     int tn = maxTime  - tm;
//     double *tsData = new double[tn*9];
//     double *R = new double[9];
//     double *coor_mod = new double[natoms1*3];
//     double *coor_mod_template = new double[natoms1*3];
//     //for (int i=0; i<natoms1*3; i++) coor_mod_template[i]=0.0;

//     for (int ti=0; ti<tn; ti++)
//     {
//       // reset modified coordinate set
//       for (int i=0; i<natoms1*3; i++) { coor_mod[i]=0.0; coor_mod_template[i]=0.0; }
//       // get rotation matrix for second coordinate set (ubiquitin)
//       GetRotation(&(R[0]), natoms2, &(coor2_template[0]), &(coor2[(ti+tm)*natoms2*3]), 0);
//       // apply R on first coordinate set (aot), take first frame set as template for all others
//       // if (ti == 0) {
//       // 	for (int a=0; a<natoms1; a++) {
//       // 	  for (int k=0; k<3; k++) {
//       // 	    for (int l=0; l<3; l++) {
//       // 	      coor_mod_template[a*3+k] += R[k*3+l] * coor1[(ti+tm)*natoms1*3 + a*3 + l];
//       // 	    }
//       // 	  }
//       // 	}
//       // }
//       for (int a=0; a<natoms1; a++) {
// 	for (int k=0; k<3; k++) {
// 	  for (int l=0; l<3; l++) {
// 	    // apply R on aot of current time step
// 	    coor_mod[a*3+k] += R[k*3+l] * coor1[(ti+tm)*natoms1*3 + a*3 + l];
// 	    // apply R on aot of first time step
// 	    coor_mod_template[a*3+k] += R[k*3+l] * coor1[tm*natoms1*3 + a*3 + l];
// 	  }
// 	}
//       }
//       // Get rotation matrix for aot
//       GetRotation(&(tsData[ti*9]), natoms1, coor_mod_template, coor_mod, 0);
//     }

//     double* wmfts = new double[(tn-2)*3];

//     calcWmfts(tsData, wmfts, tn, invDt);
    
//     double *integral = new double[tn*3];

//     calcIntegral(integral, wmfts, dt, tn);
    
//     for (int t2=1; t2<tn-2; t2++) {
//       for (int i=0; i<3; i++) {
//      	int idx = t2*3 + i;
// 	int idx0 = (t2-1)*3 + i;
//      	double tmp = std::pow(integral[3+i] - integral[idx], 2);
// 	//#pragma omp atomic
//      	msd[idx0] += tmp;
// 	//#pragma omp atomic
//      	ctr[idx0]++;
//       }
//     }

//     delete[] tsData;
//     delete[] wmfts;
//     delete[] integral;
//     delete[] coor_mod;
//     delete[] coor_mod_template;
//     delete[] R;

//   }

//   for (int i=0; i<maxTime; i++) {
//     for (int j=0; j<3; j++) {
//       msd[i*3+j] /= (double) ctr[i*3+j];
//     }
//   }
    
//   delete[] ctr;
  
//}

