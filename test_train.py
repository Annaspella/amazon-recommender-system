import pandas as pd

#Funzione SPLIT per la creazione di test e training set,
#contenenti rispettivamente il 20% e l'80% delle reviews a nostra disposizione,
#i cui item valutti abbiano rivecuto più di 11 review ciascuno
def split(data):
    users = data['reviewerID'].unique()     #lista di tutti gli user
    items = data['asin'].unique()           #lista di tutti gli item
    test = pd.DataFrame(columns=data.columns) #prende le labels delle colonne e le usa per la test set
    train = pd.DataFrame(columns=data.columns) #prende le lables delle colonne e le usa per la train set
    test_ratio = 0.2 #frazione di dati da usare per la test set
    canc = 0         #contatore del numero di records che vengono esclusi dalla ripartizione
    for i in items:
        temp = data[data['asin'] == i] #temp: contiene le reviews in riferimeto all'i-esimo item
        n = len(temp)                  #numero di review che un item ha ricevuto

        if n <= 10 :                #tengo solo gli item con più di 11 reviews
            canc= canc+n            #conto quanti sono il numero di records che non vado a mettere in train e test
            continue

        test_size = int(test_ratio*n) #calcolo del numero di review, per ciascun item, da mettere nella test set

        temp = temp.sort_values('reviewTime').reset_index() #ordina temporalmente le review in temp, date all'i-esimo item

        temp.drop('index', axis=1, inplace=True)            #rimozione di colonne non necessarie
        temp.drop('Unnamed: 0', axis=1, inplace=True)

        dummy_test = temp.loc[n-1-test_size:]   #prende le ultime test_size review di temp, da mettere in test set

        dummy_train = temp.loc[: n-2-test_size] #prendo le prime n-test_size review di temp, da mettere in train set


        test = pd.concat([test, dummy_test])    #per l'i-esimo item, aggiunge le a test set le review
        train = pd.concat([train, dummy_train]) #per l'i-esimo item, aggiunge le a train set le review

    return test, train, canc                   #ritorna test set, training set e numero totale di review escluse
