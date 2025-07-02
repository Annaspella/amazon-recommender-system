import pandas as pd
import numpy as np
import scipy
import sys
import random as rd
import matplotlib.pyplot as plt

# Randomly select the initial centroids according to the uniform distribution for the procedure.
# The function returns a matrix containing the centroids (as many columns as clusters, as many rows as users)
# and the indices of their positions in the matrix.
def PCc(B,K,seme):
    m=B.shape[0]
    n=B.shape[1]
    Centroidi=np.array([]).reshape(n,0)
    indici= np.full(K, -1, dtype=int)
    rd.seed(seme)
    for i in range(K):
        casuale=rd.randint(0,m-1)
        indici[i]=casuale
        Centroidi=np.c_[Centroidi,B[casuale]]
    return Centroidi,indici

# This function calculates the Euclidean distance between the items and the latest identified centroids.
# It returns a matrix (with as many rows as items and as many columns as clusters),
# where the element in cell i-j is the Euclidean distance of item i from centroid j.
# Each row shows the difference of the item from all the centroids.
def Euclidea(A,Centroidi):
    EuDist=np.array([]).reshape(A.shape[0],0)
    for k in range(Centroidi.shape[1]):
        temporanea=np.sqrt(np.sum((A-Centroidi[:,k])**2,axis=1))
        EuDist=np.c_[EuDist,temporanea]
    return EuDist

# This function returns the label indicating the membership of an item to a cluster.
# The vector will have as many elements as there are items.
# The indices range from 0 to K, for example, 0 indicates the first cluster.
# It is calculated by finding the minimum distance among the centroids, that is, the minimum per row.
def Etichette(DistMat):
    C=np.argmin(DistMat,axis=1)
    return C

# This function updates the centroids after defining the clusters.
# The new centroid is calculated as the mean of the values within the cluster.
# If a cluster is empty, the previous centroid is kept.
# It returns a matrix with the new centroids (as many columns as clusters, as many rows as users).
def Aggiornamento(A,Label,Centroidi):
    nuovi_Centroidi=np.array([]).reshape(A.shape[1],0)
    for j in range(Centroidi.shape[1]):
        if np.sum(Label==j)!=0:
            ricalcolo=np.mean(A[Label==j],axis=0)
            nuovi_Centroidi=np.c_[nuovi_Centroidi,ricalcolo]
        else:
            print("Il cluster", j, "è vuoto. Come cluster aggiornato prendiamo la media dei punti")
            ricalcolo=np.array(pd.DataFrame(Centroidi)[j])
            #ricalcolo=np.array(E.mean(axis=0))
            nuovi_Centroidi=np.c_[nuovi_Centroidi,ricalcolo]
    return nuovi_Centroidi

# Final function
# This function runs the K-Means algorithm.
# It returns the matrix of final centroids (as many rows as clusters, as many columns as users),
# the vector of labels, and the number of iterations performed.
# It works fo
def KMeansMio(X,k,seme=0,maxiter=100):
    X=np.array(X).T
    centroidi,primiindici=PCc(X,k,0) #trovo i primi centoridi
    for itr in range(maxiter): #inizio le iterazioni
        D=Euclidea(X,centroidi) #calcolo le distanze
        C=Etichette(D) #assegno gli item ai cluster
        nuovi_centroidi=Aggiornamento(X,C,centroidi) #calcolo i nuovi cluster
        if np.array_equal(centroidi,nuovi_centroidi): #verifico se sono cambiati
            break #se sono cambiati esco dal ciclo
        else:
            centroidi=nuovi_centroidi #ripeto l'iterazione partendo dai cluster nuovi
    return centroidi.T,C,itr

#Funzione per suddividere la Matrice di utilità non centrata in k cluster_centers.
#train: verrà passata la Matrice di Utilità pulita non centrata.
def cluster(train,k):
    M0=pd.DataFrame([])
    M1=pd.DataFrame([])
    centers, label, itr= KMeansMio(train, k) #rrecall of the kmeans function to divide into the two clusters
    for i in range(len(label)):              #iteration over the number of items
        C0=train.iloc[:,i]                   #selection of the ith column
        if(label[i]==0):                     #if the ith item belongs to the first cluste add the column M0 to C0
            M0=pd.concat([M0,C0],axis=1)
            itemcols1 = list(M0.columns)
            items_index1 = {itemcols1[i]: i for i in range(len(itemcols1))} #assign an index to each item
        else:                               #else: if the ith item belongs to the secon cluster add column M1
            M1=pd.concat([M1,C0],axis=1)
            itemcols2 = list(M1.columns)
            items_index2 = {itemcols2[i]: i for i in range(len(itemcols2))}
    return M0,M1,items_index1,items_index2
