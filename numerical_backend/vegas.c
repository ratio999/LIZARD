#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "nrutil.h"
#include <omp.h>
#define ALPH 1.5
#define NDMX 50
#define MXDIM 50
#define TINY 1.0e-30
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)


static long idum = -12345;

void vegas(double *regn, int ndim, double (*fxn)(double *, double), int init,
		unsigned long ncall, int itmx, int nprn, double *tgral, double *sd,
		double *chi2a){

	int pool = omp_get_num_procs()-1;
	omp_set_nested(1);
	
	int N_2_level = 1; 
	int N_1_level =8; 
	
	
	double ran2(long *idum);
	void rebin(double rc, int nd, double *r, double *xin, double **xi, const int j);
 	static int it, k;	
	static int mds,nd,ndo,ng,npg;
	static double calls,dv2g,dxg,f,f2,f2b,fb,rc,ti,tsi,wgt,xjac,xn,xnd,xo;
	int *kg = (int*)malloc(ndim*sizeof(int));
	static double schi,si,swgt;
	static double **xi, **d, **di;
 	double *dt = (double*)malloc((MXDIM+1)*sizeof(double));	
	double *xin =(double*)malloc((MXDIM+1)*sizeof(double));	
	xi = malloc((MXDIM+1)*sizeof(double*));
 
	d = malloc((MXDIM+1)*sizeof(double*));
	di = malloc((MXDIM+1)*sizeof(double*));
	
	
	double *domain = (double*)malloc(ndim*sizeof(double));
	double *dx = (double*)malloc((MXDIM+1)*sizeof(double));
	
	if (init <= 0) { 
		mds=ndo=1; 
		#pragma omp parallel for num_threads(N_2_level) schedule(static) shared(ndim, xi)
		for (int j=0;j<ndim;j++){ 
			xi[j] = malloc((NDMX+1)*sizeof(double));
			xi[j][0]=1.0; 
		}
	}
	if (init <= 1) si=swgt=schi=0.0;
	if (init <= 2) { 
		nd=NDMX; 
		ng=1;
		if (mds != 0) { 
			ng=(int)pow(ncall/2.0+0.25,1.0/ndim);
			mds=1;
			if ((2*ng-NDMX) >= 0) {
				mds = -1;
				npg=ng/NDMX+1;
				nd=ng/npg;
				ng=npg*nd;
			}
		}
		k=1;
		for (int i=0;i<ndim;i++) k *= ng;
		npg=IMAX(ncall/k,2);
		calls=(double)npg * (double)k;
		dxg=1.0/ng;
		dv2g=1.0;
		//free(k);
		#pragma omp parallel for num_threads(N_2_level) schedule(dynamic) shared(ndim, dxg) reduction(*:dv2g)
		for (int i=0;i<ndim;i++) dv2g *= dxg;
		
		dv2g=SQR(calls*dv2g)/npg/npg/(npg-1.0);
		xnd=nd;
		dxg *= xnd;
		xjac=1.0/calls;
		
		#pragma omp parallel for num_threads(N_2_level) schedule(static) reduction(*: xjac) 
		for (int j=0;j<ndim;j++) {
			dx[j]=regn[j+ndim]-regn[j];
			domain[j] = regn[j]; //new arr for trying reducing false sharing
			xjac *= dx[j];
		}
	 	
	 	double *r = (double*)malloc((NDMX+1)*sizeof(double));		
		if (nd != ndo) {
			int threshold = IMAX(nd,ndo);
		        #pragma omp parallel for shared(threshold) 
			for (int i=0;i<threshold;i++) r[i]=1.0;
			
			#pragma omp parallel for num_threads(N_2_level) schedule(static)
			for (int j=0;j<ndim;j++) {
				d[j] = malloc((NDMX+1)*sizeof(double));
				di[j] = malloc((NDMX+1)*sizeof(double));
				rebin(ndo/xnd,nd,r,xin,xi,j);
			}	
			ndo=nd;
			free(r);
			r=NULL;
		} //start from assigning xi[j] for avoiding false sharing. Then KEEP the parallel region on j<ndim. and maybe on k. I think not on for(;;)
		
		if (nprn >= 0) {
			printf("%s: ndim= %3d ncall= %8.0f\n",
					" Input parameters for vegas",ndim,calls);
			printf("%28s it=%5d itmx=%5d\n"," ",it,itmx);
			printf("%28s nprn=%3d ALPH=%5.2f\n"," ",nprn,ALPH);
			printf("%28s mds=%3d nd=%4d\n"," ",mds,nd);
			for (int j=0;j<ndim;j++) {
				printf("%30s xl[%2d]= %11.4g xu[%2d]= %11.4g\n",
						" ",j,regn[j],j,regn[j+ndim]);
			}
		}
	}
	for (it=0;it<itmx;it++) {
		printf("%d \n", it);
		ti=tsi=0.0;
		#pragma omp parallel for num_threads(N_2_level) schedule(static)
		for (int j=0;j<ndim;j++) {
			kg[j]=1;
			for (int i=0;i<nd;i++) d[j][i]=di[j][i]=0.0; //i and j have been inverted!!
		}
		
		int *ia = (int*)malloc(NDMX*sizeof(int)); 
		 
		double *di_1d = (double*)malloc(ndim*sizeof(double));
		double *d_1d = (double*)malloc(ndim*sizeof(double));
			
		for (;;) {
			double fb=f2b=0.0;
			#pragma omp parallel for num_threads(N_2_level) schedule(static)
			for (int j=0; j<ndim;j++){
				di_1d[j] = 0.0;
				d_1d[j] = 0.0;
			}
			#pragma omp parallel for num_threads(N_1_level) schedule(static) shared(xi, npg, xjac, dxg, xnd) private(wgt, k) reduction(+: fb, f2b, di_1d[:ndim], d_1d[:ndim])
			for (k=0;k<npg;k++) {
				double wgt=xjac;
				int *iia = (int*)malloc(ndim*sizeof(int));
				double *x = (double*)malloc(NDMX*sizeof(double));
								

				#pragma omp parallel for num_threads(N_2_level) schedule(static) shared(ndim, dxg, xnd) private(xo, xn, rc) reduction(*: wgt)
				for (int j=0;j<ndim;j++) {
					double xn;
					#pragma omp critical
					{	
					xn=(kg[j]-ran2(&idum))*dxg+1.0; //on c++code it is different
					}
					double xo;
					double rc;
					iia[j]=IMAX(IMIN((int)(xn),NDMX),1);
					if (iia[j] > 1) {
						xo=xi[j][iia[j]-1]-xi[j][iia[j]-2];
						rc=xi[j][iia[j]-2]+(xn-iia[j])*xo;
					} else {
						xo=xi[j][iia[j]-1];
						rc=(xn-iia[j])*xo;
					}
					x[j]=domain[j]+rc*dx[j];
					
					wgt *= xo*xnd;
				}
				double f=wgt*(*fxn)(x, wgt);
				double f2=f*f;
				
				fb += f;
				f2b += f2;
				if (mds>=0){	
					#pragma omp parallel for num_threads(N_2_level) schedule(static) shared(mds, iia) 
					for (int j = 0; j<ndim; j++){
						di_1d[j] += f;
						d_1d[j] += f2;
					}

				} else {
					#pragma omp parallel for num_threads(N_2_level) schedule(static) shared(mds, iia)
					for (int j = 0; j<ndim; j++){
						di_1d[j] += f;
					}
				}
				if (k==npg-1) {	
					for (int j=0; j<ndim;j++) ia[j] = iia[j];
				}
			}
			#pragma omp parallel for num_threads(N_2_level) schedule(static) 
			for (int j=0;j<ndim;j++) {
				di[j][ia[j]-1] += di_1d[j];
				if (mds >= 0){
					d[j][ia[j]-1] += d_1d[j];
				}
			}
			
		

			f2b=sqrt(f2b*npg);
			f2b=(f2b-fb)*(f2b+fb);
			if (f2b <= 0.0) f2b=TINY;
			ti += fb;
			tsi += f2b;
			if (mds < 0) { 
				#pragma omp parallel for num_threads(N_2_level) schedule(static)
				for (int j=0;j<ndim;j++) d[j][ia[j]-1] += f2b;
			}
			
			
				for (k=ndim-1;k>=0;k--) {
					kg[k] %= ng;
					if (++kg[k] != 1) break;
				}
				if (k < 0) break;
		}
		tsi *= dv2g;
		wgt=1.0/tsi;
		si += wgt*ti;
		schi += wgt*ti*ti;
		swgt += wgt;
		*tgral=si/swgt;
		*chi2a=(schi-si*(*tgral))/(it-0.0001);
		if (*chi2a < 0.0) *chi2a = 0.0;
		*sd=sqrt(1.0/swgt);
		tsi=sqrt(tsi);
		#pragma omp parallel for num_threads(N_2_level) schedule(static) private(xo, xn, rc)
		for (int j=0;j<ndim;j++) { 
			double xo=d[j][0];
			double xn=d[j][1];
			d[j][1]=(xo+xn)/2.0;
			dt[j]=d[j][0];
			double rc;
			for (int i=2;i<nd;i++) {
				rc=xo+xn;
				xo=xn;
				xn=d[j][i];
				d[j][i-1] = (rc+xn)/3.0;
				dt[j] += d[j][i-1];
			}
			d[j][nd-1]=(xo+xn)/2.0;
			dt[j] += d[j][nd-1];
		}
		#pragma omp parallel for num_threads(N_2_level) schedule(static) private(rc) 
		for (int j=0;j<ndim;j++) {
			double *r = (double*)malloc(nd*sizeof(double));
			double rc=0.0;
			for (int i=0;i<nd;i++) {
				if (d[j][i] < TINY) d[j][i]=TINY;
				r[i]=pow((1.0-d[j][i]/dt[j])/
						(log(dt[j])-log(d[j][i])),ALPH);
				rc += r[i];
			}
			#pragma omp critical
			rebin(rc/xnd,nd,r,xin,xi,j);
		}
	}
}





void rebin(double rc, int nd, double *r, double *xin, double **xi, const int j)
{
	int i,k=0;
	double dr=0.0,xn=0.0,xo=0.0;
	for (i=0;i<nd-1;i++) {
		while (rc > dr)
			dr += r[(++k)-1];
		if (k > 1) xo=xi[j][k-2];
		xn=xi[j][k-1];
		dr -= rc;
		xin[i]=xn-(xn-xo)*dr/r[k-1];
	}
	for (i=0;i<nd-1;i++) xi[j][i]=xin[i];
	xi[j][nd-1]=1.0;
}

double ran2(long *idum){
	int j;
	long k;
	static long idum2=123456789;
	static long iy=0;
	static long iv[NTAB];
	double temp;

	if (*idum <= 0) { 
		
		if (-(*idum) < 1) *idum=1; 
		else *idum = -(*idum);
		
		idum2=(*idum);
		
		for (j=NTAB+7;j>=0;j--) { 
			k=(*idum)/IQ1;
			*idum=IA1*(*idum-k*IQ1)-k*IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB) iv[j] = *idum;
		}

		iy=iv[0];
	}

	k=(*idum)/IQ1; 
	
	*idum=IA1*(*idum-k*IQ1)-k*IR1;
	
	if (*idum < 0) *idum += IM1;
	
	k=idum2/IQ2;
	
	idum2=IA2*(idum2-k*IQ2)-k*IR2;
	
	if (idum2 < 0) idum2 += IM2;
	
	j=iy/NDIV;
	
	iy=iv[j]-idum2; 
	
	iv[j] = *idum; 
	
	if (iy < 1) iy += IMM1;
	
	if ((temp=AM*iy) > RNMX) return RNMX; 
	
	else return temp;
}


