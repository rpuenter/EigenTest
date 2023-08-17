#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:34:30 2023

@author: rpuenter
"""

import numpy as np
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi']  = 600
mpl.rcParams['savefig.dpi'] = 600

from math import log10, floor

def find_exp(number) -> int:
    base10 = log10(abs(number))
    return abs(floor(base10))

def plotMap(Z,n,m,title='Plot',nLevels=101):
   
    Z = np.log(Z)
    
    zmn = np.min(Z)
    zmx = np.max(Z)
    
    expMx  = abs(find_exp(zmx))
    expMn  = abs(find_exp(zmn))
    expArr = np.array([expMn,expMx])
    limArr = np.array([zmn,zmx])
    idxMx  = np.argmin(expArr)
    zlim   = abs(limArr[idxMx])
    
    levels1 = np.linspace(-zlim, 0.,num=nLevels,endpoint=True)
    levels2 = np.linspace(0., zlim,num=nLevels,endpoint=True)
    levels  = levels1+levels2
            
    nIter = np.arange(n)+1
    mSize = np.arange(m-1)+2
    nn,mm=np.meshgrid(nIter,mSize)
    
        
    fig,ax = plt.subplots(1,1)    

    im = ax.contourf(nn,mm,Z,levels=levels,cmap='seismic')
    ax.set_xlabel(r'$N_{iter}$') 
    ax.set_ylabel(r'$M_{size}$')
    ax.set_yscale('log')
    label = r'$log\left(\frac{t_{inv}}{t_{ls}}\right)$'
    fig.colorbar(im, ax=ax,label=label)
    plt.title(title)    
    return

def plotLine(Z,m,title='Plot'):
    mSize = np.arange(m-1)+2
    fig,ax = plt.subplots(1,1)    
    ax.plot(mSize,Z)
    ax.set_ylabel(r'$\frac{\left|\lambda_{mx}\right|}{\left|\lambda_{mn}\right|}$') 
    ax.set_xlabel(r'$M_{size}$')
    ax.set_yscale('log')
    plt.title(title)
    return

class solveTest:
    '''
    Base class. Not intended to be used directly
    '''
    
    def __init__(self,nIter,mSize):
        self.nIter = nIter
        self.mSize = mSize
        
    def solveInverse(self):
        
        tinv = np.zeros((self.mSize-1,self.nIter),dtype=float)
        
        for mj in np.arange(self.mSize-1)+2:
            
            self.fillMatrix(mj)
            
            # Compute time for inverse
            start = time.time()
            mInv = np.linalg.inv(self.A)
            end = time.time()
            tInvert = start-end
            
            # Compute time for matrix multiplication
            start = time.time()
            _ = np.matmul(mInv,self.b)
            end = time.time()
            tMult = start-end
            
            for ni in np.arange(self.nIter)+1: 
                tinv[mj-2,ni-1] = tInvert+float(ni)*tMult

        return tinv
    
    def solveDirect(self):
        
        tls = np.zeros((self.mSize-1,self.nIter),dtype=float)
        
        for mj in np.arange(self.mSize-1)+2:
            
            self.fillMatrix(mj)
            
            # Compute time for linear solves
            start = time.time()
            _ = np.linalg.solve(self.A,self.b)
            end = time.time()
            t = start-end
            
            for ni in np.arange(self.nIter)+1: 
                
                tls[mj-2,ni-1] = t*float(ni)

        return tls
    
    def conditionNumber(self,delta=1e-12):
        
        cn = np.zeros((self.mSize-1,),dtype=float)
        
        for mj in np.arange(self.mSize-1)+2:
            
            self.fillMatrix(mj)
            
            eigs,_ = np.linalg.eig(self.A)
            abseigs = np.abs(eigs)
            cNumber = np.max(abseigs)/(np.min(abseigs)+delta)
            cn[mj-2] = cNumber

        return cn    
    
    def fillMatrix(self,mSiz):
        self.b = np.ones((mSiz,),dtype=float)
        return 
    
class Hilbert(solveTest):
    
    def __init__(self,nIter,mSize):
        super().__init__(nIter,mSize)
        self.name = 'Hilbert '
        
    def fillMatrix(self,mSiz):
        super().fillMatrix(mSiz)
        self.A = np.zeros((mSiz,mSiz),dtype=float)
        for i in range(mSiz):
            for j in range(mSiz):
                self.A[j,i] = 1./float(i+j+1)
        return
     

class DingDong(solveTest):
    '''
    Ref : https://math.nist.gov/MatrixMarket/data/MMDELI/dingdong/dingdong.html
    '''
    
    def __init__(self,nIter,mSize):
        super().__init__(nIter,mSize)
        self.name = 'DingDong '
        
    def fillMatrix(self,mSiz):
        super().fillMatrix(mSiz)
        self.A = np.zeros((mSiz,mSiz),dtype=float)
        for i in range(mSiz):
            for j in range(mSiz):
                den = 2*(mSiz-i-j)+1
                self.A[j,i] = 1./float(den)
        return
        
class Wilkinson(solveTest):
    '''
    Ref: https://math.nist.gov/MatrixMarket/data/MMDELI/wilkinson/wilkinson.html
    '''
    
    def __init__(self,nIter,mSize,C=1.):
        super().__init__(nIter,mSize)
        self.C = C
        self.name = 'Wilkinson '
 
    def fillMatrix(self,mSiz):
        super().fillMatrix(mSiz)
        self.A = np.eye(mSiz,dtype=float)
        for i in range(mSiz):
            self.A[i,mSiz-1] = 1. 
        for i in range(mSiz-1):
            self.A[i+1,i] = -self.C    
        return
    
def test(case,n,m):
    solver = case(n,m)
    tinv   = solver.solveInverse()
    tls    = solver.solveDirect()
    cNumber = solver.conditionNumber()
    tRatio = np.divide(tinv,tls)
    plotMap(tRatio,n,m,title=solver.name+'matrices. LAPACK gsev vs Inverse')
    plotLine(cNumber,m,title=solver.name+'matrices')
    return


if __name__== "__main__":
    n = 10
    m = 500
    test(Hilbert,n,m)
    test(DingDong,n,m)
    test(Wilkinson,n,m)
    
