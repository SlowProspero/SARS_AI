# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:08:18 2023

@author: mathieu.frechin
"""


# the libraries are loaded here
import os
import argparse
import numpy as np
import pandas as pd
# if BN_basics is not found you will need to use sys.path.append(r'foo') to add the library to the path.
import bnlearn as bn
import matplotlib.pyplot as plt


def runCoroDAG(DS, axisName, figVal=0, timeThresh=0,
               posSyn=7):
    """
    bnlearn binding
    """

    posSynLogic = posSyn
    dat = {}
    predat = {}
    dfhot = {}
    dfnum = {}
    DAG = {}
    G = {}
    model = {}
    preDS = np.copy(DS)
    preDS[:, 2:posSynLogic] = np.log2(preDS[:, 2:posSynLogic])
    DS[:, 2:posSynLogic] = np.log2(DS[:, 2:posSynLogic])
    print('log2 on drymass')

    timeList = DS[:, 1]
    DS = np.delete(DS, 1, 1)
    preDS = np.delete(preDS, 1, 1)
    posSynLogic = int(posSynLogic-1)
    for iCond in range(3):
        if iCond == 0:
            pos = np.logical_and(DS[:, 0] == iCond, timeList > timeThresh)
            DS[pos, posSynLogic] = 0
        elif iCond == 1:
            pos = np.logical_and(pos == False, timeList > timeThresh)
            pos[DS[:, 0] == 4] = False
        elif iCond == 2:
            pos = np.logical_and(DS[:, 0] == 4, timeList > timeThresh)
        dat[iCond] = pd.DataFrame(
            np.array(DS[pos, 1:], dtype=np.int32), columns=axisName[2:])
        predat[iCond] = pd.DataFrame(
            np.array(preDS[pos, 1:], dtype=np.float32), columns=axisName[2:])
        dfhot[iCond], dfnum[iCond] = bn.df2onehot(dat[iCond], verbose=4)
        DAG[iCond] = bn.structure_learning.fit(dfnum[iCond])
        DAG[iCond] = bn.parameter_learning.fit(
            DAG[iCond], dfnum[iCond], methodtype='maximumlikelihood')
        plt.figure(iCond+figVal)
        # print('figure '+str(int(iCond+figVal))+' shows the DAG of mock against infected')
        # G[iCond] = bn.plot(DAG[iCond],interactive=True)
        model[iCond] = bn.parameter_learning.fit(DAG[iCond], dfnum[iCond])
        model[iCond] = bn.independence_test(
            model[iCond], dfnum[iCond], test='chi_square', prune=True)
    return iCond+figVal+1, DAG, model, G, dfhot, dfnum, predat


def averageStats(model, cstat, chiScr, testn, Max=False, nBoot=10000):
    for iMod in range(3):

        source = np.array(model[iMod]['independence_test'][['source']])
        target = np.array(model[iMod]['independence_test'][['target']])
        currstatval = np.copy(
            np.array(model[iMod]['independence_test'][[testn]], dtype=np.uint16))
        for i in range(len(source)):
            key = source[i]+'__'+target[i]
            key = str(key)
            try:
                tmp = chiScr[iMod][key]
                if Max:
                    if tmp < currstatval[i]:
                        chiScr[iMod][key] = currstatval[i]
                else:
                    chiScr[iMod][key] = tmp+(currstatval[i]/nBoot)
                cstat[iMod][key] = cstat[iMod][key]+1
            except:
                if Max:
                    chiScr[iMod][key] = currstatval[i]
                else:
                    chiScr[iMod][key] = (currstatval[i]/nBoot)
                cstat[iMod][key] = 1
    return chiScr, cstat


def start_bn(filename):

    # reading here the data
    ds = pd.read_csv(filename)
    ds = ds.drop('Unnamed: 0', axis=1)

    # 10k bootstrapping set here
    nBoot = 5
    itest0 = {}
    itest1 = {}
    itest2 = {}

    # getting labels
    axisName = list(ds.columns.values)

    # looping over nBoot bootstrap number.
    for iboot in range(nBoot):

        print(str(iboot)+'\n')
        # generating the current dataset using random sampling with replacement
        BootDS = ds.sample(n=ds.shape[0], replace=True)
        BootDS = np.array(BootDS)

        # running bayesian inference for this iteration
        Fnum, DAG, model, G, dfhot, dfnum, predat = runCoroDAG(
            BootDS, axisName)
        # running independence Chi squared tests
        model[0] = bn.independence_test(
            model[0], dfnum[0], test='chi_square', prune=True)
        model[1] = bn.independence_test(
            model[1], dfnum[1], test='chi_square', prune=True)
        model[2] = bn.independence_test(
            model[2], dfnum[2], test='chi_square', prune=True)

        # storing all edges
        if iboot == 0:
            mat0 = np.copy(np.array(model[0]['adjmat'], dtype=np.uint16))
            mat1 = np.copy(np.array(model[1]['adjmat'], dtype=np.uint16))
            mat2 = np.copy(np.array(model[2]['adjmat'], dtype=np.uint16))
        else:
            mat0 = mat0 + \
                np.copy(np.array(model[0]['adjmat'], dtype=np.uint16))
            mat1 = mat1 + \
                np.copy(np.array(model[1]['adjmat'], dtype=np.uint16))
            mat2 = mat2 + \
                np.copy(np.array(model[2]['adjmat'], dtype=np.uint16))

        # specifically storing chi squared scores for faster browsing
        itest0[nBoot] = model[0]['independence_test']
        itest1[nBoot] = model[1]['independence_test']
        itest2[nBoot] = model[2]['independence_test']
        testn = 'chi_square'
        if iboot == 0:
            chiScr = {}
            cstat = {}
            nchiScr = {}
            for iMod in range(3):
                chiScr[iMod] = {}
                cstat[iMod] = {}
                nchiScr[iMod] = {}

        # storing the averaged edge-specific chi-squared score
        chiScr, cstat = averageStats(model, cstat, chiScr, testn, Max=False, nBoot=nBoot)

    # Frequencies of the graph edges
    normM00 = mat0/nBoot
    normM11 = mat1/nBoot
    normM22 = mat2/nBoot
    print("Output:")
    print(normM00)
    print(normM11)
    print(normM22)

def main(filename):
    start_bn(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    filename = args.filename
    if not os.path.isfile(filename):
        print("The file doesn't exist.")
        raise SystemExit(1)
    filename = args.filename

    main(filename)
