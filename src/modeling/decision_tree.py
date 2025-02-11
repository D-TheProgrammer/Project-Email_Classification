############################  Importation des bibliothèques nécessaires  ############################
import sys
import os
# Dossier 'src' au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# Données d'entrainement, de validation et de test
from preparation.preparation_data import X_train_vec, X_test_vec, X_validation_vec, y_train, y_test, y_validation, arrondir

# Modèle d'Arbre de décision
from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV

# Analyse et test des performances des modèles
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

#Pour écrire dans un fichier
import json

#################################  Modèle d'Arbre de décision  ######################################    
def main():  
    print("\n\n###################  Modèle d'Arbre de décision  ###################\n")

    # 1) Création du modèle d'Arbre de décision avec la graine 42
    decision_tree = DecisionTreeClassifier(criterion='entropy', max_features='sqrt', random_state=42)

    # 2) Entrainement du modèle avec les données X_train_vec et y_train
    decision_tree.fit(X_train_vec, y_train)

    # 3) Calcul de l'Accuracy avec les données de test
    y_test_prediction = decision_tree.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_test_prediction)
    test_accuracy = arrondir(test_accuracy)
    print("\nAccuracy des données de test : ", test_accuracy)

    # # 4) Initialisation des hyperparamètres
    # parametres_test = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}
    #     # Paramètres par défaut :
    #         # criterion : gini
    #         # max_depth : None
    #         # min_samples_split : 2
    #         # min_samples_leaf : 1
    #         # max_features : None

    #     # Meilleurs paramètres :
    #         # criterion : entropy
    #         # max_depth': None
    #         # min_samples_split : 2
    #         # min_samples_leaf : 1
    #         # max_features : sqrt

    # # 5) Recherche des meilleures hyperparamètres
    # decision_tree_temp = GridSearchCV(DecisionTreeClassifier(random_state=42), parametres_test, cv=2)
    # decision_tree_temp.fit(X_train_vec, y_train)

    # # 6) Affichage des meilleurs paramètres
    # print("\nMeilleurs paramètres :\n", decision_tree_temp.best_params_)

    # # 7) Création d'un nouveau modèle avec les meilleures hyperparamètres
    # decision_tree_final = decision_tree_temp.best_estimator_

    # 8) Calcul de l'Accuracy avec les données de validation
    y_validation_prediction = decision_tree.predict(X_validation_vec)
    decision_tree_validation_accuracy = accuracy_score(y_validation, y_validation_prediction)
    decision_tree_validation_accuracy = arrondir(decision_tree_validation_accuracy)
    print("Accuracy des données de validation : ", decision_tree_validation_accuracy)

    # 9) Calcul de la Precision
    decision_tree_precision = precision_score(y_validation, y_validation_prediction, pos_label='spam')
    decision_tree_precision = arrondir(decision_tree_precision)
    print("Précision des données de validation : ", decision_tree_precision)

    # 10) Calcul du Recall
    decision_tree_recall = recall_score(y_validation, y_validation_prediction, pos_label='spam')
    decision_tree_recall = arrondir(decision_tree_recall)
    print("Recall des données de validation : ", decision_tree_recall)

    # 11) Calcul de la Matrice de confusion
    decision_tree_matrice_confusion = confusion_matrix(y_validation, y_validation_prediction)
    print("Matrice de confusion des données de validation : ", decision_tree_matrice_confusion)

    # 12) Affichage d'un graphique de la matrice de confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(decision_tree_matrice_confusion, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.xlabel('Valeurs prédites (Y Prédictions)')
    plt.ylabel('Valeurs réels (Y Validation)')
    plt.title('Matrice de confusion de la classification d\'email \nDecision Tree')
    plt.show()

    # 13) Calcul du F1-Score
    decision_tree_f1_score = f1_score(y_validation, y_validation_prediction, pos_label='spam')
    decision_tree_f1_score = arrondir(decision_tree_f1_score)
    print("F1-Score des données de validation : ", decision_tree_f1_score)

    # 14) Calcul de la courbe ROC et de l'AUC
    taux_faux_positif, taux_vrai_positif, seuil = roc_curve(y_validation, decision_tree.predict_proba(X_validation_vec)[:, 1], pos_label='spam')
    decision_tree_auc = auc(taux_faux_positif, taux_vrai_positif)
    decision_tree_auc = arrondir(decision_tree_auc)
    print("AUC des données de validation : ", decision_tree_auc)

    # 15) Affichage de la courbe ROC
    plt.figure()
    plt.plot(taux_faux_positif, taux_vrai_positif, lw=2, label='Courbe ROC')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC Decision Tree')
    plt.legend(loc="lower right")
    plt.show()


    #Sauvegarde des resultat pour traitement
    decision_tree_resultats = {
        "validation_accuracy": decision_tree_validation_accuracy,
        "precision": decision_tree_precision,
        "recall": decision_tree_recall,
        "f1_score": decision_tree_f1_score,
        "auc": decision_tree_auc
    }

    chemin_dossier_results = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    chemin_fichier = os.path.join(chemin_dossier_results, 'decision_tree_resultats.json')
    # Créer le dossier 'results' s'il n'existe pas
    if not os.path.exists(chemin_dossier_results):
        os.makedirs(chemin_dossier_results)

    with open(chemin_fichier, 'w') as f:
        json.dump(decision_tree_resultats, f)
        
    print("Résultats sauvegardes dans le dossier 'results' ")
    
if __name__ == "__main__":
    main()