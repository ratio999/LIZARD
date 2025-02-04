#include <stdio.h>
#include <math.h>
#include "nrutil.h"
#include <omp.h>
#define ALPH 0.5 
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
#define BETA 0.75

//static long idum = -12345;

static void a_vegas(void *realf, double *regn, int ndim, double (*fxn)(void *, double *, double, int), int init,
		unsigned long ncall, int itmx, int nprn, double *tgral, double *sd,
		double *chi2a, long idum){
	
	double ran2(long *idum);
	void rebin(double rc, int nd, double r[], double xin[], double xi[MXDIM+1][NDMX+1], const int j);
	
	static int i,l,it,j,k,mds,nd,ndo,ng,*npg, ia[MXDIM+1],kg[MXDIM+1];
	static double *calls, *dv2g,dxg,f,f2,f2b,fb,rc,ti,tsi,wgt,*xjac,xn,xnd,xo;
	static double calls_start, dv2g_start, xjac_start;
	static int nev;
	static double d[NDMX+1][MXDIM+1],di[NDMX+1][MXDIM+1],dt[MXDIM+1],
	dx[MXDIM+1][NDMX+1], r[NDMX+1],x[MXDIM+1],xi[MXDIM+1][NDMX+1],xin[NDMX+1];
	static double schi,si,swgt;
	
	if (init <= 0) { 
		mds=ndo=1; 
		for (j=0;j<ndim;j++) xi[j][0]=1.0; 
	}
	if (init <= 1) si=swgt=schi=0.0;
	if (init <= 2) { 
		nd=NDMX; 
		ng=1;
		if (mds != 0) { 
	
			ng=(int)pow(ncall/4.0+0.25,1.0/ndim); //Nst on paper: 4-->2 for classic vegas
	
			mds=1;
			if ((2*ng-NDMX) >= 0) {
				mds = -1;
				nev=ng/NDMX+1;
				nd=ng/nev;
				ng=nev*nd;
			}
		}
		for (k=1,i=0;i<ndim;i++) k *= ng;
		nev=IMAX(ncall/k,2); //nev in paper
		calls_start=(double)nev * (double)k;
		dxg=1.0/ng; //delta yst in paper
		xnd=nd;
		dxg *= xnd;
		xjac_start=1.0/calls_start;
		
		for (l = 0; l<ng; l++){
			npg[l] = nev;
			calls[l] = calls_start;
			dv2g[l] = 1;
			xjac[l] = xjac_start;
			for (j=0;j<ndim;j++) {
				dv2g[l] *= dxg
				dx[j]=(regn[j+ndim]-regn[j])/ng; //interval of hyptercube l
				xjac[l] *= dx[j];
			}
			dv2g[l]=SQR(calls[l]*dv2g[l])/npg[l]/npg[l]/(npg[l]-1.0);
			
		}
		
		if (nd != ndo) { 
			for (i=0;i<IMAX(nd,ndo);i++) r[i]=1.0;
			for (j=0;j<ndim;j++) rebin(ndo/xnd,nd,r,xin,xi, j);
			ndo=nd;
		}
		
		if (nprn >= 0) {
			printf("%s: ndim= %3d ncall= %8.0f\n",
					" Input parameters for vegas",ndim,calls);
			printf("%28s it=%5d itmx=%5d\n"," ",it,itmx);
			printf("%28s nprn=%3d ALPH=%5.2f\n"," ",nprn,ALPH);
			printf("%28s mds=%3d nd=%4d\n"," ",mds,nd);
			for (j=0;j<ndim;j++) {
				printf("%30s xl[%2d]= %11.4g xu[%2d]= %11.4g\n",
						" ",j,regn[j],j,regn[j+ndim]);
			}
		}
	}
	for (it=0;it<itmx;it++) {
		//printf("Iteration number %d \n", it);

		ti=tsi=0.0;
		for (j=0;j<ndim;j++) {
			kg[j]=1;
			for (i=0;i<nd;i++) d[i][j]=di[i][j]=0.0;
		}
		for (;;) {
			total_dl = 0.0;
			for (l=0;l<ng;l++){
			    til[l] = tsil[l] = wl[l] = 0.0;
				fbl[l] = f2bl[l] = 0.0;
				for (k=0;k<npg[l];k++) {
					wgt[l]=xjac[l];
					for (j=0;j<ndim;j++) {
						//if (it>=3 ) printf(" it: %d, k: %d , j: %d \n", it, k, j );
			
						xn=(kg[j]-ran2(&idum))*dxg+1.0; //on c++code it is different
						//if (it>=3 ) printf(" it: %d, k: %d , j: %d , xn : %f \n", it, k, j, xn);
						
						ia[j]=IMAX(IMIN((int)(xn),NDMX),1);
						if (ia[j] > 1) {
							xo=xi[j][ia[j]-1]-xi[j][ia[j]-2];
							rc=xi[j][ia[j]-2]+(xn-ia[j])*xo;
						} else {
							xo=xi[j][ia[j][l]-1];
							rc=(xn-ia[j])*xo;
						}
						x[j]=regn[j] + (l+rc)*dx[j];
						wgt[l] *= xo*xnd;
					}
					//if (it>=3 ) printf(" it: %d, k: %d , FUNCIN \n", it, k );
					f=wgt*(*fxn)(realf, x, wgt[l], ndim);
					//if (it>=3 ) printf(" it: %d, k: %d ,  FUNCout \n", it, k );
					f2=f*f;
					fbl[l] += f;
					f2bl[l] += f2;
					for (j=0;j<ndim;j++) {
						di[ia[j]-1][j] += f;
						if (mds >= 0){
							d[ia[j]-1][j] += f2;
						}
					}
				}
			til[l] = fbl[l];
			f2bl[l] = sqrt(f2bl[l]*npg[l]);
			f2bl[l] = (f2bl[l] - fbl[l])*(f2bl[l] + fbl[l]);
			if (f2bl[l] >= 0.0) f2bl[l] = TINY;
			
		 	tsil[l] = f2bl[l] * dv2g[l];
			
			total_dl += pow(tsil[l], BETA);
			wl[l] = 1/tsil[l];
			si += wl[l]*til[l];
			swgt += wl[l];
			schi += til[l]*til[l]*wl[l];	
			
			//if (mds < 0) { 
			//	for (j=0;j<ndim;j++) d[ia[j]-1][j] += f2b; //stratifies sampling happens here
			//}
			
			}

			for (l=0; l<ng; l++){
				npg[l]=2+(int)(pow(tsil[l],BETA)/total_dl);
			
			}
			
			for (k=ndim-1;k>=0;k--) {
				kg[k] %= ng;
				if (++kg[k] != 1) break;
			}
			if (k < 0) break;
		}
		
		*tgral=si/swgt;
		
		*chi2a=(schi-si*(*tgral))/(it-0.0001);
		if (*chi2a < 0.0) *chi2a = 0.0;
		
		*sd=sqrt(1.0/swgt);
		
		tsi=sqrt(tsi);
		
		for (j=0;j<ndim;j++) { 
			xo=d[0][j];
			xn=d[1][j];
			d[0][j] = (7*xo + xn)/8.0;
			dt[j]=d[0][j];
			for (i=2;i<nd;i++) {
				rc= xo + 6*xn;
				xo=xn;
				xn=d[i][j];
				d[i-1][j] = (rc+xn)/8.0;
				dt[j] += d[i-1][j];
			}
			d[nd-1][j]=(xo+7*xn)/8.0;
			dt[j] += d[nd-1][j];
		}
		for (j=0;j<ndim;j++) {
			rc=0.0;
			for (i=0;i<nd;i++) {
				if (d[i][j] < TINY) d[i][j]=TINY;
				r[i]=pow((1.0-d[i][j]/dt[j])/
						(log(dt[j])-log(d[i][j])),ALPH);
				rc += r[i];
			}
		
			rebin(rc/xnd,nd,r,xin,xi, j);
		
		}
	}
}





void rebin(double rc, int nd, double r[], double xin[], double xi[MXDIM+1][NDMX+1], const int j)
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

double ran2(long *idumn){
	int j;
	long k;
	static long idum2=123456789;
	static long iy=0;
	static long iv[NTAB];
	double temp;

	if (*idumn <= 0) { 
		
		if (-(*idumn) < 1) *idumn=1; 
		else *idumn = -(*idumn);
		
		idum2=(*idumn);
		
		for (j=NTAB+7;j>=0;j--) { 
			k=(*idumn)/IQ1;
			*idumn=IA1*(*idumn-k*IQ1)-k*IR1;
			if (*idumn < 0) *idumn += IM1;
			if (j < NTAB) iv[j] = *idumn;
		}

		iy=iv[0];
	}

	k=(*idumn)/IQ1; 
	
	*idumn=IA1*(*idumn-k*IQ1)-k*IR1;
	
	if (*idumn < 0) *idumn += IM1;
	
	k=idum2/IQ2;
	
	idum2=IA2*(idum2-k*IQ2)-k*IR2;
	
	if (idum2 < 0) idum2 += IM2;
	
	j=iy/NDIV;
	
	iy=iv[j]-idum2; 
	
	iv[j] = *idumn; 
	
	if (iy < 1) iy += IMM1;
	
	if ((temp=AM*iy) > RNMX) return RNMX; 
	
	else return temp;
}

