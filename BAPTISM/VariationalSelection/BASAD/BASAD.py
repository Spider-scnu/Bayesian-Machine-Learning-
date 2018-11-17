import sys
from numpy import *
import numpy as np
import sympy
from scipy.stats import invgauss,norm
#import math

def basad(X, Y, K, delta, sig, nburn, niter, nsplit, kk):
    # get the size of sample 
    n,m = X.shape
    p = m-1
    
    # setting up the \sigma_1 and \sigma_0
    s1 = sig * max([math.log(n),0.01*p**(2 + delta)/n])
    s0 = (0.1 * sig)/n
    
    # solve c
    cp = choicep(K,p)
    
    # every batchsize of valsize
    vsize = floor(m/nsplit)
    vsize = int(vsize)
    G = np.dot(X.transpose(),X)
    # the shape and rate of gamma distribution
    p1 = 10**(-4)
    p2 = 10**(-4)
    
    pr = float(cp[0])/p
    print('cp:',cp,'s1:',s1,'s0',s0,'pr:',pr,p1)
    
    B = zeros([niter+nburn+1,m])
    B_mle = zeros([niter+nburn+1,m])
    Z = zeros([niter+nburn+1,m])
    prob = zeros([niter+nburn+1,m])
    var = zeros([niter+nburn+1,1])
    var[0] = sig
    # print(sig)
    prob[0,:] = ones([1,m])
    Z[0,0] = 1
    for brnm in range(nburn+niter):
        
        brn = brnm+1
        
        print('iter:',brn)
        B[brn,:], B_mle[brn,:] = gib_beta(X, Y, B[brn-1,:],\
                                          B_mle[brn-1,:],var[brn-1], Z[brn-1,:],\
                                          n,m,p,s1,s0,vsize,nsplit,G,p1,p2)
        var[brn] = gib_sig(X, B[brn,:], B_mle[brn,:], Z[brn-1,:],n,m,\
                           p,s1,s0,vsize,nsplit,G,p1,p2,Y)
        Z[brn,:], prob[brn,:] = gib_z(B[brn,:], var[brn], pr,n,\
                                      m,p,s1,s0,vsize,nsplit,G,p1,p2,Y)
        Z[brn,0] = 1
    return Z, B, prob, B_mle


def choicep(K,p):
    x = sympy.symbols('x')
    # zhengtai de fangcha
    #cp = sympy.solve(x-K+1/2*invgauss.ppf(0.9,1)*pow(x*(1-x/p),1/2))
    cp = sympy.solve(x-K+1.2816*pow(x*(1-x/p),1/2))
    return cp
    
def gib_beta(X,Y,B,B_mle,sig,Z,n,m,p,s1,s0,vsize,nsplit,G,p1,p2):
    B = B.reshape(m,1)
    B_mle = B_mle.reshape(m,1)
    T1 = Z*s1 + (ones([1,m])-Z)*s0
    vec = np.arange(vsize)
    
    for s in range(nsplit):
        svec = []
        Csvec = []
        svec = np.arange(s*vsize, (s+1)*vsize)
        Csvec = np.arange(1,m+1)
        Csvec[svec] = False
        COV = G[s*vsize:(s+1)*vsize,s*vsize:(s+1)*vsize] + (np.diag(np.reciprocal(T1[0,svec])))
        values, vectors = np.linalg.eig(COV)
        COVsq = mat(vectors) * mat(np.diag(np.reciprocal(pow(values,1/2)))) * mat(vectors.transpose())
        if vsize == m:
            #print('Trained by all sample:')
            B[svec] = COVsq * (COVsq*(X[:,svec]).transpose() *\
                               Y + pow(sig,1/2)[0]*np.random.randn(vsize,1))
            B_mle[svec] = COVsq * (COVsq*(X[:,svec]).transpose() * Y)
        else:
            #print('Trained by mini-batch:')
            B_Csvec = (B[Csvec!=bool(0)]).reshape(m-len(svec),1)
            B[svec] = COVsq * (COVsq * ((np.dot(X[:,svec].transpose() , Y) -\
                                        np.dot(G[s*vsize:(s+1)*vsize,Csvec!=bool(0)],B_Csvec))) +\
                                       pow(sig,1/2)*np.random.randn(len(svec),1))
            B_mle[svec] = COVsq * (COVsq * (np.dot(X[:,svec].transpose() , Y) -\
                                            np.dot(G[s*vsize:(s+1)*vsize, Csvec!=bool(0)],(B_Csvec))))
    if m > nsplit*vsize:

        #print('Citation II')
        svec = []
        Csvec = []
        COVsq = []
        svec = np.arange(nsplit * vsize, m)
        Csvec = np.arange(1, m + 1)
        Csvec[svec] = False
        
        COV = G[nsplit*vsize:m,nsplit*vsize:m] + (np.diag(np.reciprocal(T1[0,svec])))
        values, vectors = np.linalg.eig(COV)
        COVsq = mat(vectors) * mat(np.diag(np.reciprocal(pow(values,1/2)))) * mat(vectors.transpose())
        
        if vsize == m:
            #print('Train by all sample:')
            B[svec] = COVsq * (COVsq*(X[:,svec]).transpose() *\
                               Y + pow(sig,1/2)[0]*np.random.randn(vsize,1))
            B_mle[svec] = COVsq * (COVsq*(X[:,svec]).transpose() * Y)

        else:
            #print('Train by mini-batch:')
            B_Csvec = (B[Csvec!=bool(0)]).reshape(m-len(svec),1)
            B[svec] = COVsq * (COVsq * ((np.dot(X[:,svec].transpose() , Y) -\
                                        np.dot(G[nsplit*vsize:m,Csvec!=bool(0)],B_Csvec))) +\
                                       pow(sig,1/2)*np.random.randn(len(svec),1))
            B_mle[svec] = COVsq * (COVsq * (np.dot(X[:,svec].transpose() , Y) -\
                                            np.dot(G[nsplit*vsize:m, Csvec!=bool(0)],(B_Csvec))))   
    B = B.reshape(m)
    B_mle = B_mle.reshape(m)
    return B, B_mle    
    
def gib_sig(X, B, B_mle, Z, n,m,p,s1,s0,vsize,nsplit,G,p1,p2,Y):
    T1 = Z*s1 + (ones([1,m]) - Z)*s0
    a = p1 + n/2 + p/2
    B = B.reshape(m,1)
    B_mle = B_mle.reshape(m,1)
    b = p2 + (1/2) * np.dot(np.dot(B.transpose(),np.diag(np.reciprocal((T1[0])))),B)\
           + (1/2) * np.dot((Y - np.dot(X,B)).transpose(), (Y - np.dot(X,B)))
    var = 1/(np.random.gamma(a,1/b) )
    #print('b:',b,'a:',a,'var:',var,'B_mle-sum:',sum(B_mle))
    return var

def gib_z(B, sig,pr,n,m,p,s1,s0,vsize,nsplit,G,p1,p2,Y):
    B = B.reshape(m,1)
    #print('sig:',sig,'s1:',s1)
    phi_1 = norm.pdf(B,0,math.sqrt(sig*s1))
    phi_0 = norm.pdf(B,0,math.sqrt(sig*s0))
    prob = multiply(pr*phi_1,np.reciprocal(pr*phi_1 + (1-pr)*phi_0))
    tmp = np.random.rand(m,1) - prob
    Z = zeros([m,1])
    Z[tmp<0] = 1
    return Z.transpose(), prob.transpose()