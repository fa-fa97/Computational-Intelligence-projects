#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:55:38 2019

@author: fatemehf
"""

import numpy as np
def gd(inp, tar, wei, eta,itera):
    for data in range(itera):
        #activation = x*w
        activation = np.dot(inp,wei)
        #w(n+1) =w(n)+eta*x(n)*e(n)
        wei += eta*(np.dot(np.transpose(inp), tar-activation))
        print "ITERATION " + str(data)
        print wei
    # Sample input
#    activation = np.dot(np.array([[0,0,-1],[1,0,-1],[1,1,-1],[0,0,-1]]),wei)
#    print activation

 
if __name__ == '__main__':
    
    nIn = 2
    nOut = 1
    nData = 4
    x = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[1],[0],[0],[0]])
    
    x = np.concatenate((x,-0.5*np.ones((nData,1))),axis=1) #add bias input = -0/5
#    weights = -np.random.rand(nIn +1,nOut)
    
    weights = -np.ones((nIn +1,nOut))
    print(weights)
    iteration = 650

    gd(x, y, weights, 0.1,iteration)