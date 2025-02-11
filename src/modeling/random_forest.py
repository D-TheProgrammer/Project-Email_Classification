############################  Importation des bibliothèques nécessaires  ############################
import sys
import os

# Dossier 'src' au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Données d'entrainement, de validation et de test
from preparation.preparation_data import X_train_vec, X_test_vec, X_validation_vec, y_train, y_test, y_validation, arrondir

# Modèle Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Analyse et test des performances des modèles
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

#Pour écrire dans un fichier
import json

###########################  Modèle Forêt aléatoire (Random Forest)  ################################
def main():
    print("\n\n###################  Modèle Forêt aléatoire (Random Forest)  ###################\n")

    # 1) Création du modèle Random Forest avec la graine 42
    random_forest = RandomForestClassifier(
        n_estimators=100, 
        max_depth=None,  
        min_samples_split=2,  
        min_samples_leaf=1,  
        max_features='sqrt', 
        random_state=42       
    )

    # 2) Entrainement du modèle avec les données X_train_vec et y_train
    random_forest.fit(X_train_vec, y_train)

    # 3) Calcul de l'Accuracy avec les données de test
    y_test_prediction = random_forest.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_test_prediction)
    test_accuracy = arrondir(test_accuracy)
    print("Accuracy des données de test : ", test_accuracy)

    # 4) Initialisation des hyperparamètres
    #Avec tous les paramètre que nous avons testé 
    #parametres_test = {'n_estimators': [100, 200, 500], "max_depth" : [None, 10, 20], "min_samples_split" : [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}

        # n_estimators : 100
        # max_depth : None
        # min_samples_split : 2
        # min_samples_leaf : 1
        # max_features : sqrt

    # 5) Recherche des meilleures hyperparamètres version GridSearch
    #random_forest_temp = GridSearchCV(RandomForestClassifier(random_state=42), parametres_test, cv=1)
    #random_forest_temp.fit(X_train_vec, y_train)

    # 6) Affichage des meilleurs paramètres version GridSearchCV
    #print("\nMeilleurs paramètres :\n", random_forest_temp.best_params_)

    # 7) Création d'un nouveau modèle avec les meilleures hyperparamètres
    #random_forest_final = random_forest_temp.best_estimator_

    # 8) Calcul de l'Accuracy avec les données de validation
    # y_validation_prediction = random_forest_final.predict(X_validation_vec)
    y_validation_prediction = random_forest.predict(X_validation_vec)
    random_forest_validation_accuracy = accuracy_score(y_validation, y_validation_prediction)
    random_forest_validation_accuracy = arrondir(random_forest_validation_accuracy)
    print("\nAccuracy des données de validation : ", random_forest_validation_accuracy)

    # 9) Calcul de la Precision
    random_forest_precision = precision_score(y_validation, y_validation_prediction, pos_label='spam')
    random_forest_precision = arrondir(random_forest_precision)
    print("Précision des données de validation : ", random_forest_precision)

    # 10) Calcul du Recall
    random_forest_recall = recall_score(y_validation, y_validation_prediction, pos_label='spam')
    random_forest_recall = arrondir(random_forest_recall)
    print("Recall des données de validation : ", random_forest_recall)

    # 11) Calcul de la Matrice de confusion
    random_forest_matrice_confusion = confusion_matrix(y_validation, y_validation_prediction)
    print("Matrice de confusion des données de validation : ", random_forest_matrice_confusion)

    # 12) Affichage d'un graphique de la matrice de confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(random_forest_matrice_confusion, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.xlabel('Valeurs prédites (Y Prédictions)')
    plt.ylabel('Valeurs réels (Y Validation)')
    plt.title('Matrice de confusion de la classification d\'email \nRandom Forest')
    plt.show()

    # 13) Calcul du F1-Score
    random_forest_f1_score = f1_score(y_validation, y_validation_prediction, pos_label='spam')
    random_forest_f1_score = arrondir(random_forest_f1_score)
    print("F1-Score des données de validation : ", random_forest_f1_score)

    # 14) Calcul de la courbe ROC et de l'AUC
    #taux_faux_positif, taux_vrai_positif, seuil = roc_curve(y_validation, random_forest_final.predict_proba(X_validation_vec)[:, 1], pos_label='spam')
    taux_faux_positif, taux_vrai_positif, seuil = roc_curve(y_validation, random_forest.predict_proba(X_validation_vec)[:, 1], pos_label='spam')
    random_forest_auc = auc(taux_faux_positif, taux_vrai_positif)
    random_forest_auc = arrondir(random_forest_auc)
    print("AUC des données de validation : ", random_forest_auc)

    # 15) Affichage de la courbe ROC
    plt.figure()
    plt.plot(taux_faux_positif, taux_vrai_positif, lw=2, label='Courbe ROC')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC Random Forest')
    plt.legend(loc="lower right")
    plt.show()

    
    #Sauvegarde des resultat pour traitement
    random_forest_resultats = {
        "validation_accuracy": random_forest_validation_accuracy,
        "precision": random_forest_precision,
        "recall": random_forest_recall,
        "f1_score": random_forest_f1_score,
        "auc": random_forest_auc
    }

    chemin_dossier_results = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    chemin_fichier = os.path.join(chemin_dossier_results, 'random_forest_resultats.json')
    # Créer le dossier 'results' s'il n'existe pas
    if not os.path.exists(chemin_dossier_results):
        os.makedirs(chemin_dossier_results)

    with open(chemin_fichier, 'w') as f:
        json.dump(random_forest_resultats, f)
    
    print("Résultats sauvegardes dans le dossier 'results' ")

    
if __name__ == "__main__":
    main()