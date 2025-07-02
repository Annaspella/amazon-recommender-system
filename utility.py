import pandas as pd
import numpy as np

#Funzione create_utility_matrix, per la creazuione della matrice di utilità
def create_utility_matrix(data, formatizer = {'user':0, 'item': 1, 'value': 2}):

    itemField = formatizer['item']
    userField = formatizer['user']
    valueField = formatizer['value']

    userList = data.iloc[:,userField].tolist()

    itemList = data.iloc[:,itemField].tolist()

    valueList = data.iloc[:,valueField].tolist()  #tolist, converte un dataframe in una lista

    users = list(dict.fromkeys(data.iloc[:,userField]).keys())
    items = list(dict.fromkeys(data.iloc[:,itemField]).keys()) #creo una lista di item iterabili

    #set() method is used to convert any of the iterable to sequence of
    #iterable elements with dintinct element
    #Ogni volta l'ordine nella lista cambia ma non cambiano gli elementi
    #vale sia per users che per item, con dict.fromkeys restituisce la lista
    #di elementi ordinata secondo l'ordine di apparizione originario

    users_index = {users[i]: i for i in range(len(users))} #indicizza ogni user
    pd_dict = {item: [np.nan for i in range(len(users))] for item in items}
    #si ottiene così una matrice di Nan in cui verranno inseriti i valori dei rating quando rappresentati

    for i in range(0,len(data)): #per ciascuna riga del dataframe passato
        item = itemList[i]       #estrae l'item
        user = userList[i]       #estrae l'user
        value = valueList[i]     #estrae il rating assegnato
        pd_dict[item][users_index[user]] = value

    X = pd.DataFrame(pd_dict)
    X.index = users

    itemcols = list(X.columns)
    items_index = {itemcols[i]: i for i in range(len(itemcols))}
    # users_index gives us a mapping of user_id to index of user
    # items_index provides the same for items
    return X, users_index, items_index

#Funzione centered contenuta, atta alla creazione
#della matrice di utilità con dati Nan posti uguali a 0 i rating presenti centrati sulla media per item
def centered(train, irow, icol):
    utilMat = np.array(train)    #trasformo il dataset passato in array

    mask = np.isnan(utilMat)     #dummy per i valori nulli
    #Nan o valori non disponibili dell'array sono mascherati
    masked_arr = np.ma.masked_array(utilMat, mask)
    #Calcolo della media per item, quindi per colonna
    item_means = np.mean(masked_arr, axis=0)
    #I valori Nan sono sostituiti dal valore medio per item
    utilMat = masked_arr.filled(item_means)
    x = np.tile(item_means, (utilMat.shape[0],1))
    # Viene rimossa la media per item da ciascuna cella della matrice
    # I valori inizialmente Nan varranno ora 0
    utilMat = utilMat - x

    d_utilMat = pd.DataFrame(utilMat, index=irow, columns=icol)
    return d_utilMat


def noncentered(train, irow, icol):
    nuovatr = np.array(train)
    mask = np.isnan(nuovatr)
    masked_arr = np.ma.masked_array(nuovatr, mask)

    nuovatr = masked_arr.filled(0)
    # i valori Nan sono sostituiti dal valore 0
    NEWtr = pd.DataFrame(nuovatr, index=irow, columns=icol)
    return NEWtr

#Funzione pulizia contenuta che si occupa di eliminare
#righe e colonne delle matrici, in cui i valori sono tutti pari a 0

def pulizia(df):
    C = df.loc[:, (df != 0).any(axis=0)] #tiene le colonne (item) della dataframe passato dove almeno un elemento !=0
    CC  = C.loc[(C != 0).any(axis=1),:]  #tiene le righe (user) della dataframe passato dove almeno un elemento !=0
    itemcols = list(CC.columns)
    items_index = {itemcols[i]: i for i in range(len(itemcols))}
    return CC, items_index
