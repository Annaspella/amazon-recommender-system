import pandas as pd
import numpy as np
from numpy.linalg import norm


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

#####Tutte le funzioni implementate per calcolare le distanze tra item, contenute nel modulo distance.py saranno utilizzate
#    all'interno della funzione MatricePredizioni del modulo prev.py

#Funzione cosine per ottenere una matrice delle distanze coseno tra item
def cosine(train,irow):
    M = np.array(train)
    dot = np.dot(M.T,M)                              #Numeratore della funzione coseno: ottengo una matrice con
                                                     #le combinazione di moltiplicazione vettorialevettori item
    norm = np.apply_along_axis(np.linalg.norm, 0, M) #Calcolo le norme per colonna, quindi per item
    div = np.outer(norm,norm)                        # Denominatore della funzione seno:
                                                     # ottengo una matrice di combinazione di norme
    cos = dot / div                                  #Matrice delle distaanze coseno come divisione tra matrice Numeratore
                                                     # e matrice Denominatore
    d_cos = pd.DataFrame(cos, index=irow, columns=irow)
    return d_cos

#Funzione jaccard per ottenere una matrice delle distanze jaccard tra item
def jaccard(train, irow):
    X = np.array(train)
    output = np.dot(train.T,train)    #Ottengo una matrice di dimensioni pari a quella iniziale dove, in corrispondenza di ogni cella
                                      #ci sono le moltiplicazioni di un vettore di item per i restanti, se due item non hanno in comune
                                      #alcun user, il prodotto vettoriale sarà risultato 0
    items = np.array(train.index)
    for i in range(len(X[0])):       #Itero all'interno degli item
        union = 0
        inters = 0
        for j in range(len(X[0])):   #Itero all'interno degli item
            if i==j:                 #se i due item sono uguali, pongo la distanza pari a 1
                output[i][j]=1
                continue
            if output[i][j]==0:     #se i due item non hanno user in comune, pongo la distanza pari a 0
                continue
            #altrimenti
            i1 = list(items[train.iloc[:,i] !=0]) #estraggo item che hanno almeno un user in comune con l'i-esimo item
            i2 = list(items[train.iloc[:,j] !=0]) #estraggo item che hanno almeno un user in comune con il j-esimo item

            union = i1 + i2                       #unione delle due liste
            inters = intersection(i1, i2)         #intersezione delle due liste

            output[i][j] =  len(inters)/len(union) #misura di jaccard ottenuta come la lunghezza della lsita di unione diviso
                                                   #per la lunghezza della lista di intersezione

    d_jac = pd.DataFrame(output,index=irow,columns=irow)
    return d_jac

#Funzione jaccard_bags2 per ottenere una matrice delle distanze jaccard for bags tra item
def jaccard_bags2(train, irow):
    X = np.array(train)
    output = np.dot(train.T,train)      #come in jaccard
    items = np.array(train.index)
    for i in range(len(X[0])):          #itero nella lista degli item
        for j in range(len(X[0])):      #itero nella lista degli user
            if i==j:
                output[i][j]=0.5        #similarità massima quando i=j, confronto sullo stesso item
                continue
            elif output[i][j]!=0:       #se i due item hanno almeno un user in comune
                r1 = np.array(train.iloc[:,i].astype('int32'))     #lista dei rating per l'i-esimo item
                r2 = np.array(train.iloc[:,j].astype('int32'))     #lista dei rating per il j-esimo item
                l1 = [item for item, count in zip(items, r1) for i in range(count)] #ripeto item tante volte quanto i rating per i
                l2 = [item for item, count in zip(items, r2) for i in range(count)] #ripeto item tante volte quanto i rating per j
                union = l1 + l2                                 #unione delle due lsite
                inters = intersection(l1, l2)                   #intersezione delle due liste
                output[i][j] =  len(inters)/len(union)    #come per la jaccard

    d_jac_bags = pd.DataFrame(output,index=irow,columns=irow)
    return d_jac_bags

#Funzione euclidean per ottenere una matrice delle distanze euclidee tra item
def euclidean(train,irow):
    X = np.array(train)
    output = np.dot(train.T,train)          #come in jaccard e jaccard f.b.: cella pari a 0 indica che i due item non hanno user comuni
                                            #quindi non sono conforntabili in termini di similarità
    for i in range(len(X[0])):
        for j in range(len(X[0])):
            if output[i][j]==0: continue    #se non sono confrontabili lascio similarità pari a 0
            else:
                   output[i][j] = np.sqrt(np.sum((X[:,i]-X[:,j])**2))  #altrimenti calcolo la distanza euclidea tra i due vettori i e j
    d_euc = pd.DataFrame(output, index=irow, columns=irow )
    return d_euc
