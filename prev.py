import pandas as pd
import numpy as np
import distance as dis
import utility as uty

#Funzione per calcolare la predizione del rating dato dall'u-esimo user per l'i-esimo item.
#Passaggi: -1: Prendo gli item valutati dall'utente u
#          -2: Calcolo la similarità tra gli item valutati da u e item da predire i
#          -3: Ottenuto il vettore delle similarità, calcolo la predizione pesando i rating assegnati da u agli item
#              che ha valutato. Ricordo di riscalare la predizione per la media per item poichè la matrice dei rating passata
#              è la matrice di utilità per i dati di training centrata (Mtr)
def pred(i, u, dis, rating, MatOrig, type='cosine'): #passaggio di i e u sotto forma di stringa
      usersrows = np.array(rating.index)

      itemcols = np.array(rating.columns)



      MM = np.array(rating.loc[u,:]) #vettore di rating per l'u-esimo user, colonna della matrice di utilità
      nonzeroind = np.nonzero(MM)[0]
      items = itemcols[nonzeroind]  #lista degli item valutati dall'u-esimo user

      similarities = []  #vettore delle similarità con altri item
      R_list = []        #vettore dei rating predetti
      for i2 in iter(items):                      #itero all'interno degli item valutati da user u
          if i2 == i: continue                    #se item da iterazione item di cui predire il rating, passo ad item successivo
          if dis.loc[i,str(i2)] <= 0: continue    #se la distanza è negativa (item dissimili), passo ad item successivo
          sim = dis.loc[i,str(i2)]                #altrimenti salvo la distanza tra i e i2 nel vettore delle similarità
          similarities.append(sim)
          R_list.append(rating.loc[u,i2])         #e salvo il rating relativo alla distanza di i da i2 nel vettore dei rating


      SIM = np.array([i for i in similarities if i > 0] or None)

      #Calcolo della media per item di i
      utilMat = np.array(MatOrig.loc[:,i])
      mask = np.isnan(utilMat)
      masked_arr = np.ma.masked_array(utilMat, mask)
      mean_i = np.mean(masked_arr, axis=0)

      if (SIM.sum() == None):    #Se il vettore delle similarità è vuoro, pongo la predizione pari alla media dell'item da predire
          return mean_i
      else:
          if type == 'euclidean': #Se la distanza è euclidea, ottengo il peso w_i come inverso della distanza e calcolo la previsione
              SIM_new = 1/SIM
              R_p = mean_i + np.array(R_list).dot(SIM_new) / SIM_new.sum()
              return R_p

          else:                  #Per tutte le altre distanze calcolo la previsione

              R_p = mean_i + np.array(R_list).dot(SIM) / SIM.sum()
              return R_p

#Funzione MatricePredizioni per calcolare la "finta" matrice di utilità per la test set da confrontare con
#la "vera" matrice di utilità della test set, per i soli item e user comuni tra test e train (altrimenti impossibilità di calcolo o confronto)

#item: lista degli item in comune tra train e tes
#user: lista degli user in comune tra train e test_train
#rating e rating2 necessario per la jaccard for bags, dove matrice di utilità per calcolare le distanze è diversa da quella per calcolare
#le previsioni. rating2 sarà la matrice di utilità ottenuta con centered mentre rating sarà la matrice di utilità
#ottenuta con noncentered
#Nel caso di distanze diverse dalla Jaccard for bags, sia rating che rating2 si pongono uguali alla matrice di utilità centrata per train
##NEw_test matrice di utilità centrata per test, ridotta ai soli user e item comuni. Necessario per vedere su quali item e user fare previsioni
#ai fini di confronto

def MatricePredizioni(item, user, rating, rating2, MatOrig, irow, New_test, type):

    #Calcolo delle diverse matrici delle distanze
    if type=='cosine':
        D = dis.cosine(rating,irow)
    if type=='jaccard':
        D = dis.jaccard(rating,irow)
    if type=='euclidean':
        D = dis.euclidean(rating,irow)
    if type=='jacforbags':
        D = dis.jaccard_bags2(rating,irow)

    #inizializzazione della matrice delle predizioni
    Mat = []

    for u in iter(user):              #itero nella lista degli user
        MM = []                       #inizializzazione vettore per le previsione sull'u-esimo user
        #print(u)
        for i in iter(item):          #itero nella lista degli item

            t = New_test.loc[u, i]   #estraggo il rating della matrice di utilità

            if t==0 : MM.append(t)   #se rating è mancante, non calcolo predizione perchè non ci serve
            else:                    #altrimenti calcolo la previsione iu e la
                 pred_ui = pred(str(i), str(u), D, rating2, MatOrig, type=type)
                 MM.append(pred_ui)  #aggiornamento del vettore delle previsione per l'u-esimo user sull'i-esimo item

        Mat.append(MM)               #aggiornamento della matrice di previsioni per l'u-esimo user

    Matrice = pd.DataFrame(Mat, index= user, columns= item)
    return Matrice

#Funzione per il calcolo dell'RMSE tra dati contenuti nella matrice di utilità per test e quelli ottenuti con la funzione MatricePredizoni
def RMSE(test, pred):
    p = np.array(pred)
    o = np.array(test)
    diff = p - o
    potenza = np.power(diff, 2)
    risultato = np.sqrt(potenza.mean())
    return risultato

def predNoCent(i, u, dis, rating, MatOrig, type='cosine'): #passaggio di i e u sotto forma di stringa
      usersrows = np.array(rating.index)

      itemcols = np.array(rating.columns)



      MM = np.array(rating.loc[u,:]) #vettore di rating per l'u-esimo user, colonna della matrice di utilità
      nonzeroind = np.nonzero(MM)[0]
      items = itemcols[nonzeroind]  #lista degli item valutati dall'u-esimo user

      similarities = []  #vettore delle similarità con altri item
      R_list = []        #vettore dei rating predetti
      for i2 in iter(items):                      #itero all'interno degli item valutati da user u
          if i2 == i: continue                    #se item da iterazione item di cui predire il rating, passo ad item successivo
          if dis.loc[i,str(i2)] <= 0: continue    #se la distanza è negativa (item dissimili), passo ad item successivo
          sim = dis.loc[i,str(i2)]                #altrimenti salvo la distanza tra i e i2 nel vettore delle similarità
          similarities.append(sim)
          R_list.append(rating.loc[u,i2])         #e salvo il rating relativo alla distanza di i da i2 nel vettore dei rating


      SIM = np.array([i for i in similarities if i > 0] or None)

      #Calcolo della media per item di i
      utilMat = np.array(MatOrig.loc[:,i])
      mask = np.isnan(utilMat)
      masked_arr = np.ma.masked_array(utilMat, mask)
      mean_i = np.mean(masked_arr, axis=0)

      if (SIM.sum() == None):    #Se il vettore delle similarità è vuoro, pongo la predizione pari alla media dell'item da predire
          return mean_i
      else:
          if type == 'euclidean': #Se la distanza è euclidea, ottengo il peso w_i come inverso della distanza e calcolo la previsione
              SIM_new = 1/SIM
              R_p = np.array(R_list).dot(SIM_new) / SIM_new.sum()
              return R_p

          else:                  #Per tutte le altre distanze calcolo la previsione

              R_p = np.array(R_list).dot(SIM) / SIM.sum()
              return R_p

#Funzione MatricePredizioni per calcolare la "finta" matrice di utilità per la test set da confrontare con
#la "vera" matrice di utilità della test set, per i soli item e user comuni tra test e train (altrimenti impossibilità di calcolo o confronto)

#item: lista degli item in comune tra train e tes
#user: lista degli user in comune tra train e test_train
#rating e rating2 necessario per la jaccard for bags, dove matrice di utilità per calcolare le distanze è diversa da quella per calcolare
#le previsioni. rating2 sarà la matrice di utilità ottenuta con centered mentre rating sarà la matrice di utilità
#ottenuta con noncentered
#Nel caso di distanze diverse dalla Jaccard for bags, sia rating che rating2 si pongono uguali alla matrice di utilità centrata per train
##NEw_test matrice di utilità centrata per test, ridotta ai soli user e item comuni. Necessario per vedere su quali item e user fare previsioni
#ai fini di confronto

def MatricePredizioniNoCent(item, user, rating, rating2, MatOrig, irow, New_test, type):

    #Calcolo delle diverse matrici delle distanze
    if type=='cosine':
        D = dis.cosine(rating,irow)
    if type=='jaccard':
        D = dis.jaccard(rating,irow)
    if type=='euclidean':
        D = dis.euclidean(rating,irow)
    if type=='jacforbags':
        D = dis.jaccard_bags2(rating,irow)

    #inizializzazione della matrice delle predizioni
    Mat = []

    for u in iter(user):              #itero nella lista degli user
        MM = []                       #inizializzazione vettore per le previsione sull'u-esimo user
        #print(u)
        for i in iter(item):          #itero nella lista degli item

            t = New_test.loc[u, i]   #estraggo il rating della matrice di utilità

            if t==0 : MM.append(t)   #se rating è mancante, non calcolo predizione perchè non ci serve
            else:                    #altrimenti calcolo la previsione iu e la
                 pred_ui = predNoCent(str(i), str(u), D, rating2, MatOrig, type=type)
                 MM.append(pred_ui)  #aggiornamento del vettore delle previsione per l'u-esimo user sull'i-esimo item

        Mat.append(MM)               #aggiornamento della matrice di previsioni per l'u-esimo user

    Matrice = pd.DataFrame(Mat, index= user, columns= item)
    return Matrice
