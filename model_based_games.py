#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 18:30:47 2018

@author: ndanneman
"""

import numpy as np
import copy

# 6 rows and 7 colums
# board is a matrix of zeros.
# one color is 1, the other is -1
# an entire game is an array of matrices, one for each step

gs = np.zeros((6,7))

# Any column with a zero at the top could be played:
def playableCols(x):
    out = np.where(x[5,:] ==0)[0]
    return out

def normalizePreds(listOfRawPreds):
    out = [i+1 for i in listOfRawPreds]
    return out


# function that takes in gs, model, inOfTurn
# evaluates each possibility, returns (normalized) predictions
    
# TODO: handle intOfTurn correctly -- reverse evaluations or something!

def modelEvaluates(currentState, intOfTurn, model, playableColums):
    playableColsList = list(playableColums)
    rawPreds = []
    for i in range(len(playableColsList)):
          colToPlay = playableColsList[i]
          # what is the index of the 'lowest' zero in the col to play
          rowToPlay = min(np.where(currentState[:,colToPlay]==0)[0])
          testBoard = copy.deepcopy(currentState)
          testBoard[rowToPlay, colToPlay] = intOfTurn
          rawPreds.append(model.predict(testBoard))
    normalizedPreds = normlizePreds(rawPreds)
    return normalizedPreds
    

# function that makes a random play
def modelChoosesPlay(currentState, intOfTurn, model):
    colToPlay = -1
    playableColumns = playableCols(currentState)
    # the board is full; this is a draw
    if playableColumns.shape[0] == 0:
        # return something...
    # if there is only one place to play, play there 
    if playableColumns.shape[0] == 1:
        colToPlay = playableColumns[0]
    # if there are multiple places to play, model evaluates each
        valsForEachOption = modelEvaluates(currentState, intOfTurn, model, playableColumns)
        # weighted choice
        colToPLay = np.choice(playableColumns, 1, valsForEachOption)
    # what is the index of the 'lowest' zero in the col to play
    rowToPlay = min(np.where(currentState[:,colToPlay]==0)[0])
    out = copy.deepcopy(currentState)
    out[rowToPlay, colToPlay] = intOfTurn
    # print out
    return out