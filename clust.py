import pandas as pd
import numpy as np
import scipy
import sys
import random as rd
import matplotlib.pyplot as plt

#Seleziona casualemente, secondo la distribuzione uniforme,i centroidi iniziali per la procedura.
#La funziona ritorna una matrice contenente i centroidi (tante colonne quanti sono i cluster, tante righe quanti sono gli user)
#e gli indici della posizione di questi nella matrice.
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

#Questa funzione permette di calcolare la distanza euclidea tra gli item e gli ultimi centroidi individuati.
#Ritorna un matrice (tante righe quanti sono gli item, tante colonne quanti sono i cluster),
#dove l'elemento della cella i-j è la distanza euclidea dell'item i dal centroide j.
#Per riga troviamo la differenza dell'item da tutti i centroidi.
def Euclidea(A,Centroidi):
    EuDist=np.array([]).reshape(A.shape[0],0)
    for k in range(Centroidi.shape[1]):
        temporanea=np.sqrt(np.sum((A-Centroidi[:,k])**2,axis=1))
        EuDist=np.c_[EuDist,temporanea]
    return EuDist

#Questa funzione riporta l'etichetta che indica l'appartenza di un item al cluster.
#Il vettore avrà tanti elementi quanti sono gli item.
#Gli indici vanno da 0 a K, ad esempio 0 indica il primo cluster.
#Viene calcolato trovando la distanza minima tra i centroidi, quindi il minimo per riga.
def Etichette(DistMat):
    C=np.argmin(DistMat,axis=1)
    return C

#Questa funzione esegue l'aggiornamento dei centroidi dopo aver definito i gruppi.
#Il nuovo centroide è calcolato come media dei valori interni al cluster.
#Se un cluster risulta vuoto si mantiene il centroide precedente.
#Ritorna una matrice con i nuovi centroidi (tante colonne quanti sono i cluster, tante righe quanti sono gli user).
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

#Funzione finale
#Questa funzione esegue l'algoritmo K-Means.
#Ritorna la matrice dei centroidi finali (tante righe quanti sono i cluster, tante colonne quante sono gli user),
#il vettore delle etichette e il numero di iterazioni effettuate.
#Funziona per matrici d'utilità che originariamente hanno gli item disposti per colonna.
#Se gli item dovessero essere messi per riga bisogna passare la matrice trasposta.
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
    centers, label, itr= KMeansMio(train, k) #richiamo la funzione kmeans per la suddivisione dei due cluster
    for i in range(len(label)): #itero tante volte quanti sono gli item
        C0=train.iloc[:,i] #estrazione della colonna i-esima
        if(label[i]==0): #se l'item i-esimo appartiene al primo cluster aggiungo la colonna ad M0
            M0=pd.concat([M0,C0],axis=1)
            itemcols1 = list(M0.columns)
            items_index1 = {itemcols1[i]: i for i in range(len(itemcols1))} #indicizza ogni item
        else:#se invece l'item i-esimo appartiene al secondo cluster aggiungo la colonna ad M1
            M1=pd.concat([M1,C0],axis=1)
            itemcols2 = list(M1.columns)
            items_index2 = {itemcols2[i]: i for i in range(len(itemcols2))}
    return M0,M1,items_index1,items_index2
