#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:46:30 2018

@author: ndanneman
"""
#%% 
import numpy as np
import copy


# %% Let's play some random games of connect-4

# 6 rows and 7 colums
# board is a matrix of zeros.
# one color is 1, the other is -1

# an entire game is an array of matrices, one for each step
# we eventually want a bunch of game states, with their back-stepped scores

gs = np.zeros((6,7))

# Any column with a zero at the top could be played:
def playableCols(x):
    out = np.where(x[5,:] ==0)[0]
    return out


# function that makes a random play
def makeRandomPlay(currentState, intOfTurn):
    playableColumns = playableCols(currentState)
    colToPlay = np.random.choice(playableColumns)
    # what is the index of the 'lowest' zero in the col to play
    rowToPlay = min(np.where(currentState[:,colToPlay]==0)[0])
    out = copy.deepcopy(currentState)
    out[rowToPlay, colToPlay] = intOfTurn
    print out
    return out

def whereToRandomlyPlay(currentState):
    playableColumns = playableCols(currentState)
    colToPlay = np.random.choice(playableColumns)
    # what is the index of the 'lowest' zero in the col to play
    rowToPlay = min(np.where(currentState[:,colToPlay]==0)[0])
    return (rowToPlay, colToPlay)


# helper function for checkForWinner
# takes in vector, sees if four in a row are intOfTurn
def fourInARow(npArrayVector, intOfTurn):
    counter = 0
    for i in range(len(npArrayVector)):
        if npArrayVector[i]==intOfTurn:
            counter += 1
            if counter == 4:
                return True
        else:
            counter = 0
    return False


# function that checks for a win
# note, you only need to see if the most recent play generated a win!
# probably should use info in makeRandomPlay to do that...ie what was played last

# note: we have access to the row, col just played...
def checkForWinner(currentState, rowPlayed, colPlayed, intOfTurn):
    # check for col win
    print("col")
    if rowPlayed >= 3:
        if np.all(currentState[rowPlayed-3:rowPlayed+1,colPlayed] == intOfTurn):
            return (True, intOfTurn)
    # check for row win
    print("row")
    if fourInARow(currentState[rowPlayed,:],intOfTurn):
        return(True, intOfTurn)
    # check for diagonal wins:
    # first, the easy one (down/left, up/right)    
    print("dl")
    dl = []
    for i in np.arange(1,7):
        if rowPlayed-i >= 0 and colPlayed - i >= 0:
            dl.append(currentState[rowPlayed-i, colPlayed-i])
    vec = list(reversed(dl))
    vec.append(intOfTurn * 1.0)
    print("ur")
    up = []
    for i in np.arange(1,7):
        if rowPlayed+i <= 5 and colPlayed + i <= 6:
            up.append(currentState[rowPlayed+i, colPlayed+i])
    vec += up
    if fourInARow(vec, intOfTurn):
        return(True, intOfTurn)
    # now the up/left, down/right diag:
    print("ul")
    ul = []
    for i in np.arange(1,7):
        if rowPlayed+i <= 5 and colPlayed - i >= 0:
            ul.append(currentState[rowPlayed+i, colPlayed-i])
    vec2 = list(reversed(ul))
    vec2.append(intOfTurn * 1.0)
    print("dr")
    for i in np.arange(1,7):
        if rowPlayed-i <= 0 and colPlayed +i <= 6:
            vec2.append(currentState[rowPlayed-i, colPlayed+i])
    if fourInARow(vec2, intOfTurn):
        return(True, intOfTurn)
    return (False, intOfTurn)

tg = g[0][21]
checkForWinner(tg, 0, 2, 1)

# checkForWinner casually passes spot testing.  Ought to write a proper unit test for this... TODO

# A GAME is an array of 6/7 game state, each holding one updated move beyond the previous.
# This is poor form, as these matrices are sparse, could just hold the row/col indices of moves
# However, going to push these raw matrices into TensorFlow, so might as well make them now...

# function to play full random games
    
def randomGame():
    whoseTurn = 1
    allGameStates = []
    gs = np.zeros((6,7))
    allGameStates.append(gs)
    longestGame = gs.shape[0] * gs.shape[1]
    for i in range(longestGame):
        # where to play
        nextRow, nextCol = whereToRandomlyPlay(allGameStates[i])
        # make the play
        nextState = copy.deepcopy(allGameStates[i])
        nextState[nextRow, nextCol] = whoseTurn
        print(nextState)
        # add this move to the list of game states
        allGameStates.append(nextState)
        # check for winner (after 7 turns!)
        isWinner, whoWon = checkForWinner(nextState, nextRow, nextCol, whoseTurn)
        if isWinner:
            return((allGameStates, whoWon))
        # change whose turn it is
        whoseTurn = whoseTurn *-1
    # if nobody wins, return the states and a zero
    return((allGameStates, 0))

g = randomGame()
print("winner: " + str(g[1]))
print(g[0][len(g[0])-1])    


# function that back-steps score through array of game-states
# a game is a tuple of list of game states (np arrays), and a marker for who won (or 0)
# map final score backwards through game non-linearly
# really, it just maps 1/-1 back N steps to near-zero

# returns a sequence from first weight to final weight
def backMapReward(intOfWinner, nSteps):
    exponents = np.arange(0, nSteps)
    reved = list(reversed(exponents))
    weights = list(map((lambda x: 0.9 ** x), reved) ) 
    if intOfWinner==-1:
        weights = [-1 * x for x in weights]
    return(weights)

def addRewardToGameStates(gameStateList, intOfWinner):
    
























