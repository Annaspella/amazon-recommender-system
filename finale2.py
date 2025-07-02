import pandas as pd
import numpy as np
import scipy
from numpy.linalg import norm
import test_train as tetr
import utility as uty
import distance as dis
import prev as pre
import clust as cl
import time


data = pd.read_csv('Digital_Music.csv') #solo prime n righe, passerei il file da terminale
data['reviewerID'] = data['reviewerID'].astype('str')
data['asin'] = data['asin'].astype('str')
users = data['reviewerID'].unique()     #lista di tutti gli user
items = data['asin'].unique()           #lista di tutti gli item
print("Number of users", len(users))    #numero user: 5541
print("Number of items", len(items))    #numero di item: 3568
print(data.head(n=10))
print("Numero di record nella matrice iniziale: {}".format(data.shape[0]))


#Funzione split contenuta nel modulo tetr, per la divisione dei dati in test e train set
test, train, canc = tetr.split(data)

test_m = test[['reviewerID','asin','overall']]
train_m = train[['reviewerID','asin','overall']]
perc1 = test_m.shape[0]/(data.shape[0]-canc)
perc2 = train_m.shape[0]/(data.shape[0]-canc)
print("Numero di record esclusi, per numerosità non sufficiente: {}".format(canc))
print("Numero di record nella test set: {}, Percentuale di record nella test set: {}".format(test_m.shape[0],perc1))
print("Numero di record nella train set: {}, Percentuale di record nella train set: {}".format(train_m.shape[0],perc2))

#Funzione create_utility_matrix contenuta nel modulo utility.py
XXtr, i1tr, i2tr = uty.create_utility_matrix(train_m)
countnantr = XXtr.isnull().sum().sum()
dimtr = XXtr.shape[0]*XXtr.shape[1]
pnantr = (countnantr/dimtr)*100
print("Percentuale di Nan nella training set: {}%".format(pnantr))

XXte, i1te, i2te = uty.create_utility_matrix(test_m)
countnante = XXte.isnull().sum().sum()
dimte = XXte.shape[0]*XXte.shape[1]
pnante = (countnante/dimte)*100
print("Percentuale di Nan nella test set: {}%".format(pnante))

#Funzione centered contenuta nel modulo utility, atta al centramento dei dati
#intorno alla media per item e al trattamento dei valori Nan
Ctr = uty.centered(XXtr, i1tr, i2tr)  #centro la matrice di utilità per training
print(Ctr)                            #'''''''''''''''''''''''''''''''  test
#[5401 rows x 1715 columns]

Cte = uty.centered(XXte, i1te, i2te) #centro la matrice di utilità per test
print(Cte)
#[3575 rows x 1715 columns]


#Funzione pulizia contenuta nel modulo utility.py
#Necessaria nel caso dei dati centrati, quando un rating è pari al valor medio infatti assume
#valor zero e, nel caso sia l'unico rating per un item, il vettore dei rating per item consisterà di soli zeri

#Eliminare gli item che non hanno nemmeno un rating,
#(pulizia matrice necessaria per calcolare la norma nel caso della distanza coseno)
#Inoltre item senza nemmeno un rating inutili ai nostri fini previsivi
#poichè impossibile calcolare una misura di distanza per valutare la similarità
#con altri item

#Si pulisce anche per riga. Vale los tesso concetto,
#dove un user non ha dato alcun rating agli item presenti non è informativo
#per valutare la similarità tra item

Mtr, new_irowtr = uty.pulizia(Ctr) #pulizia della matrice di utilità per il training set
print(Mtr)
#[5394 rows x 1683 columns]
peritr= (1 - Mtr.shape[1]/Ctr.shape[1])*100
perutr= (1 - Mtr.shape[0]/Ctr.shape[0])*100
print("Percentuale di item persi con operazione di pulizia training:{}%".format(peritr))
print("Percentuale di user persi con operazione di pulizia training:{}%".format(perutr))


Mte, new_irowte = uty.pulizia(Cte) #pulizia della matrice di utilità per il test set
print(Mte)
#[3387 rows x 1479 columns]
perite= (1 - Mte.shape[1]/Cte.shape[1])*100
perute= (1 - Mte.shape[0]/Cte.shape[0])*100
print("Percentuale di item persi con operazione di pulizia test:{}%".format(perite))
print("Percentuale di user persi con operazione di pulizia test:{}%".format(perute))

print("Numero di user nella training set: {}".format(Ctr.shape[0]))
print("Numero di user nella training set pulita: {}".format(Mtr.shape[0]))

print("Numero di item nella training set: {}".format(Ctr.shape[1]))
print("Numero di item nella training set pulita: {}".format(Mtr.shape[1]))

print("Numero di user nella test set: {}".format(Cte.shape[0]))
print("Numero di user nella test set pulita: {}".format(Mte.shape[0]))

print("Numero di item nella test set: {}".format(Cte.shape[1]))
print("Numero di item nella test set pulita: {}".format(Mte.shape[1]))


#Calcolo degli user e item in comune tra test e train:
#così sarà possibile fare previsioni sulla test set, per item e user che
#ha in comune con la training set, col fine di valutare l'accuratezza previsiva
usersrows_te = Mte.index  #estrazione della lista di user dalla test set utility matrix pulita
itemcols_te = Mte.columns #estrazione della lista di item dalla test set utility matrix pulita

usersrows_tr = Mtr.index #lo stesso per la train set
itemcols_tr = Mtr.columns

common_elements_item = dis.intersection(itemcols_te, itemcols_tr)
common_elements_user = dis.intersection(usersrows_te, usersrows_tr)

print("Numero di item comuni a train e test{}".format(len(common_elements_item)))
print("Numero di user comuni a train e test{}".format(len(common_elements_user)))

#Funzione noncentered contenuta nel modulo utility,

##Matrice di utilità per i dati di training, contenente dati originari e Nan posti uguali a 0
#necessaria per il caclolo della distanza con Jaccard for bags
temp = uty.noncentered(XXtr, i1tr, i2tr)
Mttr, new_irowttr = uty.pulizia(temp)
#Pulizia in questo caso ritorna la stessa matrice perchè nessuna riga o colonna sarà di soli zeri
#dato che i rating non vengono riscalati per la media (0 solo se Nan).

##Matrcice di utilità per i dati di test, contenente dati originari e Nan posti uguali a 0
#necessaria per il caclolo dei cluster
temp1 = uty.noncentered(XXte, i1te, i2te)
Mtte, new_irowtte = uty.pulizia(temp1)

usersrows_tce = Mtte.index  #estrazione della lista di user dalla test set utility matrix non centrata pulita
itemcols_tce = Mtte.columns #estrazione della lista di item dalla test set utility matrix non centrata pulita

##Matrcice di utilità per i dati di test, contenente dati originari e Nan posti uguali a 0
#prendendo i soli item e user in comune tra train e test. Necessaria per il calcolo dell'RMSE
NEW = uty.noncentered(XXte, i1te, i2te)
New_test = NEW.loc[common_elements_user,common_elements_item]


#################################################Predizioni senza cluster:
#Modulo prev contenente funzione MatricePredizioni e RMSE (con relativa descrizione)
z=time.time()

P = pre.MatricePredizioni(common_elements_item, common_elements_user, Mtr, Mtr, XXtr, new_irowtr, New_test, type='cosine')
R1 = pre.RMSE(New_test, P)
print('\n Coseno: Ora stampo predizioni e rating da test set\n')
print(P)
print(New_test)
print(R1)
z1=time.time()
ztime=z1-z

x=time.time()
P = pre.MatricePredizioni(common_elements_item, common_elements_user, Mtr, Mtr, XXtr, new_irowtr, New_test, type='jaccard')
R2 = pre.RMSE(New_test, P)
print('\n Jaccard: Ora stampo predizioni e rating da test set\n')
print(P)
print(New_test)
print(R2)
x1=time.time()
xtime=x1-x

y=time.time()
P = pre.MatricePredizioni(common_elements_item, common_elements_user, Mttr, Mtr, XXtr, new_irowttr, New_test, type='jacforbags')
R3 = pre.RMSE(New_test, P)
print('\n Jac for bags: Ora stampo predizioni e rating da test set\n')
print(P)
print(New_test)
y1=time.time()
ytime=y1-y

r=time.time()
P = pre.MatricePredizioni(common_elements_item, common_elements_user, Mtr, Mtr, XXtr, new_irowtr, New_test, type='euclidean')
R4 = pre.RMSE(New_test, P)
print('\n Euclidea: Ora stampo predizioni e rating da test set\n')
print(P)
print(New_test)
r1=time.time()
rtime=r1-r

print("RMSE a confronto per le singole distanze. Coseno:{}, Jaccard:{}, Jaccard for bags:{},Euclidea:{}".format(R1,R2,R3,R4))

print('Tempo per Coseno:{}'.format(ztime))
print('Tempo per Jaccard:{}'.format(xtime))
print('Tempo Jac for bags:{}'.format(ytime))
print('Tempo euclidea:{}'.format(rtime))


########################################################Predizioni con cluster:
#Scelta arbitraria di k=2 perchè con k=3, valore ottenuto tramite funzione per arrivare al k ottimale,
#risultavano cluster di numerosità estremamente ridotta
#Modulo clust contentente KMeansMio e cluster(con relativa descrizione)
k=2
centers, label, itr= cl.KMeansMio(Mttr, k)

Cl0,Cl1,ic1,ic2= cl.cluster(Mttr,k)
print("Numero di item nel cluster 1: {}".format(Cl0.shape[1]))

print("Numero di item nel cluster 2: {}".format(Cl1.shape[1]))

#Calcolo degli user e item in comune tra test non centrata e i due cluster:
#così sarà possibile fare previsioni sulla test set, per item e user che
#ha in comune con i rispettivi cluster, col fine di valutare l'accuratezza previsiva
usersrows_tr1 = Cl0.index #estrazione della lista di item del primo cluster
itemcols_tr1 = Cl0.columns #estrazione della lista di user del primo cluster

usersrows_tr2 = Cl1.index #stessa cosa per il secondo cluster
itemcols_tr2 = Cl1.columns

common_elements_item1 = dis.intersection(itemcols_tce, itemcols_tr1)
common_elements_user1 = dis.intersection(usersrows_tce, usersrows_tr1)

common_elements_item2 = dis.intersection(itemcols_tce, itemcols_tr2)
common_elements_user2 = dis.intersection(usersrows_tce, usersrows_tr2)

##Matrcice di utilità per i dati di test, contenente dati originari e Nan posti uguali a 0
#prendendo i soli item e user in comune tra i due cluster e test. Necessaria per il calcolo dell'RMSE
NEW = uty.noncentered(XXte, i1te, i2te)

New_test1 = NEW.loc[common_elements_user1,common_elements_item1]
New_test2 = NEW.loc[common_elements_user2,common_elements_item2]

####################################Euclidea e cluster:
#Modulo prev contenente funzione MatricePredizioniNoCent e RMSE (con relativa descrizione)
tE0=time.time()
P1 = pre.MatricePredizioniNoCent(common_elements_item1, common_elements_user1, Cl0, Cl0, XXtr, ic1, New_test1, type='euclidean')
RC1 = pre.RMSE(New_test1, P1)
P2 = pre.MatricePredizioniNoCent(common_elements_item2, common_elements_user2, Cl1, Cl1, XXtr, ic2, New_test2, type='euclidean')
RC2 = pre.RMSE(New_test2, P2)
R_tot=RC1+RC2
tE1=time.time()
t_tot=tE1-tE0

print('\n Euclidea: Ora stampo predizioni e rating da test set per il cluster 1\n')
print(P1)
print(New_test1)


print('\n Euclidea: Ora stampo predizioni e rating da test set  per il cluster 2\n')
print(P2)
print(New_test2)


print('Tempo euclidea cluster:{}'.format(t_tot))
print('RMSE totale dell euclidea cluster: {}'.format(R_tot))

#################################Coseno e cluster
tC0=time.time()
P_C = pre.MatricePredizioniNoCent(common_elements_item1, common_elements_user1, Cl0, Cl0, XXtr, ic1, New_test1, type='cosine')
R_C = pre.RMSE(New_test1, P_C)
P_C1 = pre.MatricePredizioniNoCent(common_elements_item2, common_elements_user2, Cl1, Cl1, XXtr, ic2, New_test2, type='cosine')
R_C1 = pre.RMSE(New_test2, P_C1)
R_tot1=R_C+R_C1
tC1=time.time()
t_tot1=tC1-tC0

print('\n Cosine: Ora stampo predizioni e rating da test set per il cluster 1\n')
print(P_C)
print(New_test1)

print('\n Cosine: Ora stampo predizioni e rating da test set per il cluster 2\n')
print(P_C1)
print(New_test2)


print('Tempo cosine cluster:{}'.format(t_tot1))
print('RMSE totale del cosine cluster: {}'.format(R_tot1))



################################Jaccard e cluster
tJ0=time.time()
P_J = pre.MatricePredizioniNoCent(common_elements_item1, common_elements_user1, Cl0, Cl0, XXtr, ic1, New_test1, type='jaccard')
R_J = pre.RMSE(New_test1, P_J)
P_J1 = pre.MatricePredizioniNoCent(common_elements_item2, common_elements_user2, Cl1, Cl1, XXtr, ic2, New_test2, type='jaccard')
R_J1 = pre.RMSE(New_test2, P_J1)
R_tot2=R_J+R_J1
tJ1=time.time()
t_tot2=tJ1-tJ0

print('\n Jac: Ora stampo predizioni e rating da test set per il cluster 1\n')
print(P_J)
print(New_test1)
print(R_J)

print('\n Jac: Ora stampo predizioni e rating da test set per il cluster 2\n')
print(P_J1)
print(New_test2)
print(R_J1)

print('Tempo Jaccard cluster: {}'.format(t_tot2))
print('RMSE totale del cosine cluster: {}'.format(R_tot2))
