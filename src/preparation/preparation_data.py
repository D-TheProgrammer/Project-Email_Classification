############################  Importation des bibliothèques nécessaires  ############################

# Préparation des données
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer  # Vectorisation
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
import matplotlib.pyplot as plt
import os
nltk.download('stopwords')


###################################  Préparation des données  #######################################
print("\n###################  Préparation des données  ###################\n")


# Fonction de nettoyage des données
def nettoyage(data):
    data.loc[:, 'Message'] = data['Message'].str.replace(r'\t+', ' ', regex=True)  # Suppression des tabulations
    data.loc[:, 'Message'] = data['Message'].str.replace(r'\s+', ' ', regex=True)  #supprimer les espaces inutiles (double et autres)
    data = data.dropna(subset=['Message'])  # Suppression des lignes où la colonne "Message" est vide (NaN)
    data.loc[:, 'Message'] = data['Message'].str.lower()  # lower() permet de mettre tout le texte en minuscule
    data.loc[:, 'Message'] = data['Message'].str.replace(r'[^\w\s]', '', regex=True)   # Suppression des caractères spéciaux
    data.loc[:, 'Message'] = data['Message'].str.replace(r'\d+', '', regex=True)  # Suppression des chiffres, '', regex
    
    #Non utilisé pour cause de manque de puissance machine
    #data.loc[:, 'Message'] = data['Message'].apply(lambda x: supression_stopword(x))     #supression des stopwords
    #data.loc[:, 'Message'] = data['Message'].apply(lambda x: radicalisation(x))     # Radicalisation
    
    
    data['Longueur'] = data['Message'].apply(lambda x: len(x.split())) #longueur des messages (en mots)
    data = data[data['Longueur'] <= 2000] # Suppression des valeurs aberrantes
    return data


#Fonction de suppression des stopword
def supression_stopword(texte):
    texte = [mot for mot in texte.split() if mot.lower() not in stopwords.words('english')]
    return " ".join(texte)


#Fonction de radicalisation
def radicalisation(texte):
    radicalisation = SnowballStemmer("english")
    mots = [radicalisation.stem(mot) for mot in texte.split()]
    return " ".join(mots)


# Fonction de séparation de la variable à prédire et de la variable de prédiction
def separation_data(data):
    X = data['Message']  # Le contenu textuel de chaque email
    y = data['Categorie']  # La catégorie d'email : spam ou ham (non-spam)
    return X, y

# Fonction de rééquilibrage des données d'entrainement pour avoir autant de Spam que de Ham
def reequilibrage(X_train, y_train):
    X_train = pd.DataFrame(X_train)  # Convertion des données d'entrainement en DataFrame
    reequilibrage = RandomOverSampler(random_state=42)
    X_train, y_train = reequilibrage.fit_resample(X_train, y_train)
    return X_train, y_train

# Fonction pour arrondir un nombre à deux chiffres après la virgule
def arrondir(valeur):
    valeur = round(valeur, 2)
    return valeur


# Fonction pour tracer les histogrammes
def tracer_histogrammes(data):
    print('\n\n----------------------------------------------------------------')
    print("Plot d'histogrammes avec longueur de mail for chaque label")
    
    # Tracer l'histogramme par catégorie
    for categorie, groupe in data.groupby('Categorie'):
        plt.hist(
            groupe['Longueur'],
            bins=50,
            alpha=0.5,
            label=categorie
        )

    plt.suptitle('Répartitions des longueurs des emails par catégorie', y=1.05, fontsize=14)
    plt.ylabel('Fréquence apparition de mots', fontsize=12) 
    plt.xlabel('Nombre de mot de l\'email', fontsize=12)  
    plt.legend(title="Catégorie")
    plt.tight_layout()
    plt.show()


# 1) Importation et stockage des données d'entrainement dans un dataframe
chemin_train = os.path.join('data', 'raw', 'train.txt')
data_train = pd.read_csv(chemin_train, delimiter="\t", header=None, names=["Categorie", "Message"], on_bad_lines='skip')
print("Jeu de données d'entrainement après importation :\n", data_train.head())

# 2) Importation et stockage des données de validation dans un dataframe
chemin_validation = os.path.join('data', 'raw', 'validation.txt')
data_validation = pd.read_csv(chemin_validation, delimiter="\t", header=None, names=["Categorie", "Message"], on_bad_lines='skip')
print("\n\nJeu de données de validation après importation :\n", data_validation.head())




# 3) On applique la fontion de nettoyage sur les données d'entrainement et de validation
data_train = nettoyage(data_train)
data_validation = nettoyage(data_validation)
print("\n\nJeu de données d'entrainement après nettoyage :\n", data_train.head())
print("\n\nJeu de données de validation après nettoyage :\n", data_validation.head())

#l'email avec le plus grand nombre de mots
max_longueur = data_train['Longueur'].max()
min_longueur = data_train['Longueur'].min()

# Nombre de mails ayant le maximum de mots
emails_with_max = data_train[data_train['Longueur'] == max_longueur].shape[0]
emails_with_min = data_train[data_train['Longueur'] == min_longueur].shape[0]
print(f"\n\nLe maximum de mots dans un email avant vectorisation est : {max_longueur} mots.")
print(f"Le nombre de mails avec le maximum de mots avant vectorisation est : {emails_with_max}.")
print(f"Le minimum de mots dans un email avant vectorisation est : {min_longueur} mots.")
print(f"Le nombre de mails avec le minimum de mots avant vectorisation est : {emails_with_min}.")


# Histogrammes pour le jeu d'entraînement
tracer_histogrammes(data_train) 


# 4) On applique la fonction de séparation de la variable à prédire et de la variable de prédiction
X, y = separation_data(data_train)
X_validation, y_validation = separation_data(data_validation)
print("\n\nNombre d'email spam et ham (non-spam) des données d'entrainement :\n", y.value_counts())
print("\n\nNombre d'email spam et ham (non-spam) des données de validation :\n", y_validation.value_counts())



# 5) Séparation des données en valeur d'entraînement (70%) et de test (30%)
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

# 6) On applique la fonciton de rééquilibrage des données d'entrainement
X_train, y_train = reequilibrage(X_train_temp, y_train_temp)
print("\n\nRééquilibrage des données d'entrainement pour avoir autant de Spam que de Ham.")

# Affichage de la taille des données d'entrainement et de test
print("\n\nTaille de l'ensemble d'entraînement :", X_train.shape[0], "\n", y_train.value_counts())
print("\nTaille de l'ensemble de test :", X_test.shape[0], "\n", y_test.value_counts())
print("\nTaille de l'ensemble de validation :", X_validation.shape[0], "\n", y_validation.value_counts())

# 7) Vectorisation du texte avec la méthode TfidfVectorizer() et utilisation des stop_words
vectorisation = TfidfVectorizer(stop_words='english')

# Tf (Term Frequency) : nombre de fois qu'un mot apparait dans un email / total des mots dans l'email.
# idf (Inverse Document Frequency) : log( (total des email) / (1 + nombre d'email avec le mot) ).
# Tfidf : Tf * idf.

# Utilisation de la vectorisation sur les données d'entrainement, de validation et de test
X_train_vec = vectorisation.fit_transform(X_train["Message"])
X_test_vec = vectorisation.transform(X_test)
X_validation_vec = vectorisation.transform(X_validation)

# Affichage de la taille des données d'entrainements après la vectorisation
print("\n\nTaille des données d'entrainement après la vectorisation :")
print("Taille de l'ensemble d'entraînement :", X_train_vec.shape)
print("Taille de l'ensemble de test :", X_test_vec.shape)
print("Taille de l'ensemble de validation :", X_validation_vec.shape)
#print(vectorisation.get_feature_names_out())
print("Taille du vocabulaire vectorisé:", len(vectorisation.get_feature_names_out()))