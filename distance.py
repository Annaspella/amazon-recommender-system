import pandas as pd
import numpy as np
from numpy.linalg import norm


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

##### All the functions implemented to calculate distances between items, contained in the distance.py module, will be used
#    within the MatricePredizioni function of the prev.py module

# Cosine function to obtain a matrix of cosine distances between items
def cosine(train,irow):
    M = np.array(train)
    dot = np.dot(M.T,M)                              # Numerator of the cosine function: obtain a matrix with
                                                     # combinations of vector multiplication between item vectors
    norm = np.apply_along_axis(np.linalg.norm, 0, M) # Calculate norms for each column, i.e., for each item
    div = np.outer(norm,norm)                        # Denominator of the cosine function:
                                                     # obtain a matrix combining norms
    cos = dot / div                                  # Matrix of cosine distances as division between numerator matrix
                                                     # and denominator matrix
    d_cos = pd.DataFrame(cos, index=irow, columns=irow)
    return d_cos

# Jaccard function to obtain a matrix of Jaccard distances between items
def jaccard(train, irow):
    X = np.array(train)
    output = np.dot(train.T,train)    # Obtain a matrix of the same size as the input where each cell
                                      # contains the vector multiplications of one item by the others; if two items share no user,
                                      # the dot product will be 0
    items = np.array(train.index)
    for i in range(len(X[0])):       # Iterate through the items
        union = 0
        inters = 0
        for j in range(len(X[0])):   #Itero all'interno degli item
            if i==j:                # if the two items are the same, set distance to 1
                output[i][j]=1
                continue
            if output[i][j]==0:     # if the two items share no users, set distance to 0
                continue
            #altrimenti
            i1 = list(items[train.iloc[:,i] !=0]) # extract items that have at least one user in common with item i
            i2 = list(items[train.iloc[:,j] !=0]) # extract items that have at least one user in common with item j

            union = i1 + i2                       # union of the two lists
            inters = intersection(i1, i2)         # intersection of the two lists

            output[i][j] =  len(inters)/len(union) # Jaccard measure obtained as length of intersection divided by length of union

    d_jac = pd.DataFrame(output,index=irow,columns=irow)
    return d_jac

# Jaccard_bags2 function to obtain a matrix of Jaccard distances for bags between items
def jaccard_bags2(train, irow):
    X = np.array(train)
    output = np.dot(train.T,train)      #same as in jaccard
    items = np.array(train.index)
    for i in range(len(X[0])):          #iterate over the list of items 
        for j in range(len(X[0])):      #iterate over the list of users
            if i==j:
                output[i][j]=0.5        #maximum similarity when i=j, comparison on the same item
                continue
            elif output[i][j]!=0:       #if the two items share at least one user
                r1 = np.array(train.iloc[:,i].astype('int32'))     # list of ratings for item i
                r2 = np.array(train.iloc[:,j].astype('int32'))     # list of ratings for item j
                l1 = [item for item, count in zip(items, r1) for i in range(count)] # repeat item as many times as the rating for i
                l2 = [item for item, count in zip(items, r2) for i in range(count)] # repeat item as many times as the rating for j
                union = l1 + l2                                 # union of the two lists
                inters = intersection(l1, l2)                   # intersection of the two lists
                output[i][j] =  len(inters)/len(union)    #same as for jaccard

    d_jac_bags = pd.DataFrame(output,index=irow,columns=irow)
    return d_jac_bags

# Euclidean function to obtain a matrix of Euclidean distances between items
def euclidean(train,irow):
    X = np.array(train)
    output = np.dot(train.T,train)          # same as in jaccard and jaccard for bags: cell = 0 means the two items share no common users,
                                            # so they are not comparable in similarity terms
    for i in range(len(X[0])):
        for j in range(len(X[0])):
            if output[i][j]==0: continue   # if not comparable, leave similarity at 0
            else:
                   output[i][j] = np.sqrt(np.sum((X[:,i]-X[:,j])**2))  # otherwise calculate Euclidean distance between vectors i and j
    d_euc = pd.DataFrame(output, index=irow, columns=irow )
    return d_euc
