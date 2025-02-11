############################  Importation des bibliothèques nécessaires  ############################
import sys
import os

# Dossier 'src' au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Données d'entrainement, de validation et de test
from preparation.preparation_data import X_train_vec, X_test_vec, X_validation_vec, y_train, y_test, y_validation, arrondir

# Modèle SVM
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

# Analyse et test des performances des modèles
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

#Pour écrire dans un fichier
import json

############################  Modèle Support Vector Machines (SVM)  #################################

def main():
    print("\n\n###################  Modèle Support Vector Machines (SVM)  ###################\n")

    # 1) Création du modèle SVM avec la graine 42
    svm = LinearSVC(dual=True, random_state=42, max_iter=20000)

    # 2) Entrainement du modèle avec les données X_train_vec et y_train
    svm.fit(X_train_vec, y_train)

    # 3) Calcul de l'Accuracy avec les données de test
    y_test_prediction = svm.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_test_prediction)
    test_accuracy = arrondir(test_accuracy)
    print("Accuracy des données de test : ", test_accuracy)

    # 4) Initialisation des hyperparamètres
    parametres_test = {'C': [0.01, 0.1, 1, 10], 'loss': ['hinge', 'squared_hinge'], 'penalty': ['l2']}
        # C : 1.0
        # loss : 'squared_hinge'
        # penalty : 'l2'

    # 5) Recherche des meilleures hyperparamètres
    svm_temp = GridSearchCV(LinearSVC(dual=True, random_state=42, max_iter=20000), parametres_test, cv=3)
    svm_temp.fit(X_train_vec, y_train)

    # 6) Affichage des meilleurs paramètres
    print("\nMeilleurs paramètres :\n", svm_temp.best_params_)

    # 7) Création d'un nouveau modèle avec les meilleures hyperparamètres
    svm_final = svm_temp.best_estimator_

    # 8) Calibration du modèle final pour pouvoir appliquer la courbe ROC et l'AUC
    svm_final_calibre = CalibratedClassifierCV(svm_final, method='sigmoid', cv='prefit')
    svm_final_calibre.fit(X_validation_vec, y_validation)

    # 9) Calcul de l'Accuracy avec les données de validation
    y_validation_prediction = svm_final_calibre.predict(X_validation_vec)
    svm_validation_accuracy = accuracy_score(y_validation, y_validation_prediction)
    svm_validation_accuracy = arrondir(svm_validation_accuracy)
    print("\nAccuracy des données de validation : ", svm_validation_accuracy)

    # 10) Calcul de la Precision
    svm_precision = precision_score(y_validation, y_validation_prediction, pos_label='spam')
    svm_precision = arrondir(svm_precision)
    print("Précision des données de validation : ", svm_precision)

    # 11) Calcul du Recall
    svm_recall = recall_score(y_validation, y_validation_prediction, pos_label='spam')
    svm_recall = arrondir(svm_recall)
    print("Recall des données de validation : ", svm_recall)

    # 12) Calcul de la Matrice de confusion
    svm_matrice_confusion = confusion_matrix(y_validation, y_validation_prediction)
    print("Matrice de confusion des données de validation : ", svm_matrice_confusion)

    # 13) Affichage d'un graphique de la matrice de confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(svm_matrice_confusion, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.xlabel('Valeurs prédites (Y Prédictions)')
    plt.ylabel('Valeurs réels (Y Validation)')
    plt.title('Matrice de confusion de la classification d\'email \nSVM')
    plt.show()

    # 14) Calcul du F1-Score
    svm_f1_score = f1_score(y_validation, y_validation_prediction, pos_label='spam')
    svm_f1_score = arrondir(svm_f1_score)
    print("F1-Score des données de validation : ", svm_f1_score)

    # 15) Calcul de la courbe ROC et de l'AUC
    y_validation_probabilite = svm_final_calibre.predict_proba(X_validation_vec)[:, 1]
    taux_faux_positif, taux_vrai_positif, seuil = roc_curve(y_validation, y_validation_probabilite, pos_label='spam')
    svm_auc = auc(taux_faux_positif, taux_vrai_positif)
    svm_auc = arrondir(svm_auc)
    print("AUC des données de validation : ", svm_auc)

    # 16) Affichage de la courbe ROC
    plt.figure()
    plt.plot(taux_faux_positif, taux_vrai_positif, lw=2, label='Courbe ROC')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC , SVM')
    plt.legend(loc="lower right")
    plt.show()
    
    # Sauvegarde des résultats
    svm_resultats = {
        "validation_accuracy": svm_validation_accuracy,
        "precision": svm_precision,
        "recall": svm_recall,
        "f1_score": svm_f1_score,
        "auc": svm_auc
    }

    chemin_dossier_results = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    chemin_fichier = os.path.join(chemin_dossier_results, 'svm_resultats.json')
    # Créer le dossier 'results' s'il n'existe pas
    if not os.path.exists(chemin_dossier_results):
        os.makedirs(chemin_dossier_results)

    with open(chemin_fichier, 'w') as f:
        json.dump(svm_resultats, f)
    
    print("Résultats sauvegardes dans le dossier 'results' ")
    
    
if __name__ == "__main__":
    main()