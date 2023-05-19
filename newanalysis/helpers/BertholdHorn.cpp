// =========================================================
// This class calculates superpositioning rotation matrix
// for two sets of atom coordinates.
// 
// 
// 										  Gregor Neumayr 2006
// ==========================================================

#include <math.h>
#include <cmath>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include "BertholdHorn.h"

using namespace std;


#define ROTATE(a,i,j,k,l) g 		= a[i][j]; \
						  h 		= a[k][l]; \
						  a[i][j]	= g-s*(h+g*tau); \
						  a[k][l]	= h+s*(g-h*tau);


// ----------------------------------------------------------
// calculates the eigenvectors (v) and eigenvalues (d) of
// a real 4x4 matrix.
// 
// from numerical recipes. changed to support 4x4
// instead of nxn matrices.
// ----------------------------------------------------------
bool Jacobi(double a[4][4], double d[], double v[4][4], int debug)
{
	int j,iq,ip,i, nrot;
	double tresh,theta,tau,t,sm,s,h,g,c;
	double b[4], z[4];

	if (debug) {
	  cout << "Jacobi()" << endl;
	  cout << a[0][0] << "\t" << d[0] << "\t" << v[0][0] << endl;
	}

	for (ip = 0;ip < 4; ip++) 
	{
		for (iq = 0;iq < 4; iq++) v[ip][iq]=0.0;
		v[ip][ip]=1.0;
		b[ip]=d[ip]=a[ip][ip];
		z[ip]=0.0;
	}
	nrot=0;
	for (i=1;i<=50;i++) 
	{
		sm=0.0;
		for (ip = 0;ip < 3;ip++) 
		{
			for (iq = ip+1; iq < 4;iq++)
				sm += fabs(a[ip][iq]);
		}
		if (sm == 0.0) 
		{
			return true;
		}
		if (i < 4)
			tresh=0.2*sm/(16);
		else
			tresh=0.0;
		for (ip = 0;ip < 3; ip++) 
		{
			for (iq=ip+1; iq < 4;iq++) 
			{
				g=100.0*fabs(a[ip][iq]);
				if (i > 4 && (double)(fabs(d[ip])+g) == (double)fabs(d[ip])
					&& (double)(fabs(d[iq])+g) == (double)fabs(d[iq]))
					a[ip][iq]=0.0;
				else if (fabs(a[ip][iq]) > tresh) {
					h=d[iq]-d[ip];
					if ((double)(fabs(h)+g) == (double)fabs(h))
						t=(a[ip][iq])/h;
					else {
						theta=0.5*h/(a[ip][iq]);
						t=1.0/(fabs(theta)+sqrt(1.0+theta*theta));
						if (theta < 0.0) t = -t;
					}
					c=1.0/sqrt(1+t*t);
					s=t*c;
					tau=s/(1.0+c);
					h=t*a[ip][iq];
					z[ip] -= h;
					z[iq] += h;
					d[ip] -= h;
					d[iq] += h;
					a[ip][iq]=0.0;
					for (j=0; j <= ip-1;j++)		{ROTATE(a,j,ip,j,iq)}
					for (j=ip+1;j <= iq-1;j++)	{ROTATE(a,ip,j,j,iq)}
					for (j=iq+1;j < 4;j++)		{ROTATE(a,ip,j,iq,j)}
					for (j = 0;j < 4;j++)			{ROTATE(v,j,ip,j,iq)}
					nrot++;
				}
			}
		}
		for (ip = 0;ip < 4;ip++) {
			b[ip] += z[ip];
			d[ip]=b[ip];
			z[ip]=0.0;
		}
	}
	return false;
}


// ----------------------------------------------------------
// returns optimal superposition rotation matrix R for the 
// coordinate sets APoints and BPoints. Number of Points n  
// is equal for APoints and Bpoints.
//
// note: center of mass of the coordinates must be in the
// 		 origin
// 
// Berthold K. P. Horn (1987)
// 
// freely transcribed from 
// http://www.imaging.robarts.ca/~dgobbi/hacks/LandmarkRegistration.py
// ----------------------------------------------------------
//void GetRotation(double R[3][3], int n, double APoints[][3], double BPoints[][3])
void GetRotation(double *R_ptr, int n, double *APoints_ptr, double *BPoints_ptr, int debug)
{
  double **R = new double*[3];
  double **APoints = new double*[n];
  double **BPoints = new double*[n];
	double M[3][3], N[4][4];
	double Nvec[4][4],Nval[4], magnitude;
	int i, j, k;
  
	if (debug) cout << "GetRotation()" << endl;

	for (i=0;i<3;i++) {
	  R[i] = &(R_ptr[i*3]);
	}
	for (i=0;i<n;i++) {
	  APoints[i] = &(APoints_ptr[i*3]);
	  BPoints[i] = &(BPoints_ptr[i*3]);
	}

// calculate the 3x3 M matrix as described in Horn's paper ---------------------
	for (i = 0; i < 3; i++)
	{
    	for (j = 0; j < 3; j++)
		{
			M[j][i] = 0.0;
			for (k = 0; k < n; k++)
			{
				M[j][i] += APoints[k][i]*BPoints[k][j];
			}
		}
	}

  
	// 4x4 matrix n as described in Horn's paper
	N[0][0] =  M[0][0] + M[1][1] + M[2][2];
	N[1][1] =  M[0][0] - M[1][1] - M[2][2];
	N[2][2] = -M[0][0] + M[1][1] - M[2][2];
	N[3][3] = -M[0][0] - M[1][1] + M[2][2];

	N[0][1] = N[1][0] = M[1][2] - M[2][1];
	N[0][2] = N[2][0] = M[2][0] - M[0][2];
	N[0][3] = N[3][0] = M[0][1] - M[1][0];

	N[1][2] = N[2][1] = M[0][1] + M[1][0];
	N[1][3] = N[3][1] = M[2][0] + M[0][2];
	N[2][3] = N[3][2] = M[1][2] + M[2][1];

 
	// calculate eigenvalues and eigenvectors
	bool success = Jacobi(N, Nval, Nvec, debug);

	//if (! success) cout << "Failure!" << endl;
  
	// the rotation quaternion is the eigenvector corresponding to the largest positive eigenvalue
	double max = 0,x,y,z,w;
	int maxi = -1;
	for (i = 0; i < 4; i++)
		if (Nval[i] > max)
			max = Nval[maxi = i];

	w = Nvec[0][maxi];
	x = Nvec[1][maxi];
	y = Nvec[2][maxi];
	z = Nvec[3][maxi];

	// normalize quaternion to unit length
	magnitude = sqrt(w*w + x*x + y*y + z*z);
	w /= magnitude;  x /= magnitude;  y /= magnitude;  z /= magnitude;

  
	// transform quaternion to rotationm matrix
	R[0][0] = w*w + x*x - y*y - z*z;
	R[0][1] = 2.0 * (-w*z + x*y);
	R[0][2] = 2.0*(w*y + x*z);
	R[1][0] = 2.0 * (w*z + x*y);
	R[1][1] = w*w - x*x + y*y - z*z;
	R[1][2] = 2.0 * (-w*x + y*z);
	R[2][0] = 2.0 * (-w*y + x*z);
	R[2][1] = 2.0 * (w*x + y*z);
	R[2][2] = w*w - x*x - y*y + z*z;

	/*if (R[0][0] != R[0][0]) {
	  cout << R[0][0] << "\t" << M[0][0] << "\t" << N[0][0] << endl;
	  }*/

	delete[] R;
	delete[] APoints;
	delete[] BPoints;

}
