# Project-Email_Classification
[French] Classification d'email en spam et non-spam  
[English] Email classification into spam and non-spam

---

## SOMMAIRE / SUMMARY  
- [Présentation en français / Presentation in French](#français)  
- [Présentation en anglais / Presentation in English](#english)  
---

## __[FRANÇAIS]__  
## Introduction
Le projet **Email Classification** a pour objectif de classifier les emails en utilisant des modèles de machine learning. Nous avons mis en place une solution complète qui nécessite certaines étapes pour configurer l'environnement et faire fonctionner les scripts correctement.

## Objectifs du Projet
1. **Récolte des données** : Importation des données à partir du fichier `email.txt` et `validation.txt`
2. **Nettoyage et préparation des données** : 
   - Suppression des espaces inutiles, des tabulations, des chiffres, et des caractères spéciaux.
   - Tokenisation des données
   - Suppression des stop words (mots sans signification sémantique)
   - Réduction de la taille des documents par stemming ou lemmatization (dépendant de la puissance machine)
   - Calcul de la longueur des messages (nombre de mots) et filtrage des messages trop longs (plus de 2000 mots).
3. **Séparation des données** Division des données en ensembles d'entraînement, de test et de validation (70% train, 30% test).  
5. **Vectorisation des données** :
   - TF-IDF (Term Frequency-Inverse Document Frequency)
   - Utilisation des stopwords
6. **Entraînement de modèles** : Application de SVM, Random Forest et Arbre de Décision.
7. **Optimisation des Hyperparamètres** : Utilisation de GridSearch afin d'optimiser le modèle en fonction des paramètre disponible du modèle
8. **Évaluation des modèles** : Utilisation des métriques comme la précision, le rappel, l'accuracy, le F1-score, l'AUC, et la matrice de confusion , Roc curve
9. **Utilisation de plusieurs graphiques** : Ces graphiques permettent de voir le ratio de mots dans les données , voir la courbe ROC , voir la matrice de confusion basée sur une heatmap.
10. **Comparaison des résultats des modèles** : Utilisation des résultats précédent dans un graphique afin de pouvoir estimer le meilleur modèle, pour cela il y a fallu utiliser les résultats des modèles (précision, rappel, F1-score, AUC) enregistrés automatiquement  dans des fichiers JSON après l'entraînement de chaque modèle. Ces fichiers sont stockés dans le dossier src/results/

## Installation
### Prévention Problème lié à Matplotlib pour un environnement virtuel
Créer un environnement virtuel fonctionnel avec **matplotlib** s'est avéré difficile. Même le site officiel de matplotlib reconnaît que l'installation dans certains environnements peut poser problème. Les solutions proposées ne sont pas toujours compatibles avec toutes les configurations de machine. C'est pourquoi nous avons mis en place deux options :
1. **`main.py`** : pour exécuter le code dans un environnement standard (sans environnement virtuel).
2. **`main_si_env.py`** : pour ceux dont la machine peut supporter **matplotlib** dans un environnement virtuel.
   
### Version Non-Environnement Virtuel

Si vous ne souhaitez pas utiliser d'environnement virtuel ou si vous rencontrez des problèmes pour faire fonctionner **matplotlib** dans un environnement virtuel, vous pouvez installer les dépendances globalement sur votre machine et exécuter le projet directement. Voici les étapes :

#### 1. Se déplacer dans le répertoire du projet

Commencez par ouvrir un terminal et déplacez-vous dans le dossier du projet :

```bash
cd chemin/vers/email_classification_vfinal
```

#### 2. Installer les dépendances globalement
Exécutez la commande suivante pour installer les dépendances directement sur votre machine :
```bash
pip install -r requirements.txt
```

#### 3. Exécuter le script principal
Enfin, exécutez le script principal avec la commande :
```bash
python src/main.py
```



### Version Créer Environnement virtuel

#### 1. Se déplacer dans le répertoire du projet
Commencez par ouvrir un terminal et déplacez-vous dans le dossier du projet :

```bash
cd chemin/vers/email_classification_vfinal
```

#### 2. Créer l'environnement virtuel
Un script est inclus pour créer l'environnement virtuel. Exécutez la commande suivante :

```bash
python setup/create_env.py
```
#### 3. Installer les dépendances
Une fois l'environnement virtuel créé, installez les dépendances nécessaires avec cette commande :

```bash
python setup/install_requirements.py
```
#### 4. Activer l'environnement virtuel
Sur Windows :
```bash
email_classifier_env_vfinal\Scripts\activate
```
Sur macOS/Linux :
```bash
source email_classifier_env_vfinal/bin/activate
```
#### 5. Exécuter le script principal
Une fois l'environnement activé et les dépendances installées, vous pouvez exécuter le script principal pour lancer le projet :

```bash
python src/main_si_env.py
```


 

## Arborescence du Projet

```bash
email_classification/
├── data/
│   └── raw/
│       ├── train.txt                        # Le fichier des emails d'entraînement
│       └── validation.txt                   # Le fichier des emails de validation
├── setup_env/ 
│   ├── create_env.py                        # Script pour créer l'environnement virtuel
│   └── install_requirement.py               # Script pour installer les dépendances
├── src/
│   ├── __init__.py                          # Pour que le dossier soit un module Python
│   ├── main.py                              # Script principal pour gérer l'exécution
│   ├── modeling/
│   │    ├── __init__.py                     # Pour que le dossier soit un module
│   │    ├── random_forest.py                # Modèle Random Forest
│   │    ├── decision_tree.py                # Modèle Arbre de Décision
│   │    └── svm.py                          # Modèle SVM
│   └── preparation/
│        ├── __init__.py                     # Pour que le dossier soit un module
│        └── preparation_data.py             # Script de préparation des données
│   └── results/
│        ├── decision_tree_resultats.json    # Résultats du modèle Arbre de Décision
│        ├── random_forest_resultats.json    # Résultats du modèle Random Forest
│        └── svm_resultats.json              # Résultats du modèle SVM
├── __init__.txt                             # Pour que le dossier soit un module Python
├── requirements.txt                         # Fichier des dépendances du projet
└── email_classifier_env/                    # Environnement virtuel
```

## Dépendances
#### Manipulation de donnees
pandas 
numpy 

#### Preparation et manipulation des donnees
scikit-learn 
imbalanced-learn 
nltk 

#### Visualisation
matplotlib 
seaborn 

## Dataset
Notre dataset est vraiment conséquent il est composé de plus de 40 000 lignes de données que vous pourrez trouver ici [https://github.com/kebiri-isamdine/Classification_des_mails_spams-hams/tree/main/Data]  
Tandis que notre jeu de données pour les données de validation sont disponible ici [https://huggingface.co/datasets/mshenoda/spam-messages/tree/main]


## Résultat 
- Decision Tree : {"validation_accuracy": 0.97, "precision": 0.97, "recall": 0.95, "f1_score": 0.96, "auc": 0.97}
  
- Random Forest : {"validation_accuracy": 0.99, "precision": 0.99, "recall": 0.98, "f1_score": 0.98, "auc": 1.0} 

- SVM : {"validation_accuracy": 0.99, "precision": 0.98, "recall": 0.98, "f1_score": 0.98, "auc": 1.0} 

Pour Plus de Précision sur les Résultats Consultez les Programmes , Leur Graphiques et les Résultats dans les terminaux



---
---
## __[ENGLISH]__  
# Project-Email_Classification

## Introduction
The **Email Classification** project aims to classify emails using machine learning models. We have implemented a complete solution that requires some steps to configure the environment and make the scripts work properly.

## Project Objectives
1. **Data collection**: Import data from the `email.txt` and `validation.txt` files
2. **Data cleaning and preparation**:
- Removal of unnecessary spaces, tabs, numbers, and special characters.
- Data tokenization
- Removal of stop words (words without semantic meaning)
- Reduction of document size by stemming or lemmatization (depending on machine power)
- Calculation of message length (number of words) and filtering of messages that are too long (more than 2000 words).
3. **Data splitting** Splitting the data into training, test and validation sets (70% train, 30% test).
5. **Data vectorization** :
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Use of stopwords
6. **Model training** : Application of SVM, Random Forest and Decision Tree.
7. **Hyperparameter optimization** : Use of GridSearch to optimize the model according to the available model parameters
8. **Model evaluation** : Use of metrics such as precision, recall, accuracy, F1-score, AUC, and the confusion matrix, Roc curve
9. **Use of multiple graphs** : These graphs allow you to see the ratio of words in the data, see the ROC curve, see the confusion matrix based on a heatmap.
10. **Comparison of model results**: Using previous results in a graph to estimate the best model, for this it was necessary to use the model results (precision, recall, F1-score, AUC) automatically saved in JSON files after training each model. These files are stored in the src/results/ folder

## Installation
### Prevention Matplotlib problem for a virtual environment
Creating a working virtual environment with **matplotlib** has proven difficult. Even the official matplotlib website acknowledges that installation in some environments can be problematic. The proposed solutions are not always compatible with all machine configurations. This is why we have implemented two options:
1. **`main.py`**: to run the code in a standard environment (without a virtual environment).
2. **`main_si_env.py`**: for those whose machine can support **matplotlib** in a virtual environment.


## Installation
### Prevention Matplotlib Issue for Virtual Environment
Creating a working virtual environment with **matplotlib** has proven difficult. Even the official matplotlib website acknowledges that installing in some environments can be problematic. The solutions offered are not always compatible with all machine configurations. That is why we have implemented two options:
1. **`main.py`**: to run the code in a standard environment (without a virtual environment).
2. **`main_si_env.py`**: for those whose machine can support **matplotlib** in a virtual environment.

### Non-Virtual Environment Version

If you do not want to use a virtual environment or if you have problems getting **matplotlib** to work in a virtual environment, you can install the dependencies globally on your machine and run the project directly. Here are the steps:

#### 1. Move to the project directory

Start by opening a terminal and move to the project folder:

```bash
cd path/to/email_classification_vfinal
```

#### 2. Install dependencies globally
Run the following command to install the dependencies directly on your machine:
```bash
pip install -r requirements.txt
```

#### 3. Run the main script
Finally, run the main script with the command:
```bash
python src/main.py
```


#### 4. Activate the virtual environment
On Windows:
```bash
email_classifier_env_vfinal\Scripts\activate
```
On macOS/Linux:
```bash
source email_classifier_env_vfinal/bin/activate
```
#### 5. Run the main script
Once the environment is activated and the dependencies are installed, you can run the main script to launch the project:

```bash
python src/main_si_env.py
```

## Project Tree

```bash
email_classification/
├── data/
│ └── raw/
│ ├── train.txt # The training email file
│ └── validation.txt # The file of validation emails
├── setup_env/
│ ├── create_env.py # Script to create the virtual environment
│ └── install_requirement.py # Script to install dependencies
├── src/
│ ├── __init__.py # To make the folder a Python module
│ ├── main.py # Main script to manage the execution
│ ├── modeling/
│ │ ├── __init__.py # To make the folder a module
│ │ ├── random_forest.py # Random Forest model
│ │ ├── decision_tree.py # Decision Tree Model
│ │ └── svm.py # SVM Model
│ └── preparation/
│ ├── __init__.py # Make the folder a module
│ └── preparation_data.py # Data preparation script
│ └── results/
│ ├── decision_tree_resultats.json # Decision Tree Model Results
│ ├── random_forest_resultats.json # Random Forest Model Results
│ └── svm_resultats.json # SVM Model Results
├── __init__.txt # Make the folder a module Python
├── requirements.txt # Project dependency file
└── email_classifier_env/ # Virtual environment
```

## Dependencies
#### Data manipulation
pandas
numpy

#### Data preparation and manipulation
scikit-learn
imbalanced-learn
nltk

#### Visualization
matplotlib
seaborn

## Dataset
Our dataset is really substantial, it is composed of more than 40,000 lines of data that you can find here [https://github.com/kebiri-isamdine/Classification_des_mails_spams-hams/tree/main/Data]
While our dataset for the validation data is available here [https://huggingface.co/datasets/mshenoda/spam-messages/tree/main]

## Result

- Decision Tree : {"validation_accuracy": 0.97, "precision": 0.97, "recall": 0.95, "f1_score": 0.96, "auc": 0.97}

- Random Forest : {"validation_accuracy": 0.99, "precision": 0.99, "recall": 0.98, "f1_score": 0.98, "auc": 1.0}

- SVM : {"validation_accuracy": 0.99, "precision": 0.98, "recall": 0.98, "f1_score": 0.98, "auc": 1.0}

For More Precision on the Results Consult the Programs, Their Graphs and the Results in the terminals
