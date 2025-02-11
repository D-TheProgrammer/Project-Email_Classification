import os
import subprocess
import sys

import matplotlib.pyplot as plt
#Pour écrire dans un fichier
import json

# Entraîne un modèle en appelant le bon fichier python
def train_model(model_script):
    print("Entraînement du modèle ", model_script)

    # Ajout du chemin absolu de 'src' au PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath('src')

    # Exécution du script avec l'environnement modifié
    subprocess.check_call([os.path.join('email_classifier_env_vfinal', 'Scripts', 'python.exe'),
                           os.path.join('src', 'modeling', model_script)], env=env)


def charger_resultats(nom_fichier):
    dossier_actuel = os.path.dirname(__file__)
    chemin_fichier = os.path.join(dossier_actuel, 'results', nom_fichier)
    with open(chemin_fichier, 'r') as f:
        return json.load(f)
    
def main():
    print("Modèle à entrainer : decision_tree, random_forest, svm")
    train_model('decision_tree.py')
    train_model('random_forest.py')
    train_model('svm.py')

    random_forest_resultats = charger_resultats('random_forest_resultats.json')
    svm_resultats = charger_resultats('svm_resultats.json')
    decision_tree_resultats = charger_resultats('decision_tree_resultats.json')

    # Comparaison des résultats
    nom_modele = ["Random Forest", "SVM", "Arbre de décision"]
    performances = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]

    # Récupérer les valeurs pour chaque métrique et chaque modèle
    resultats = {
        "Random Forest": [
            random_forest_resultats["validation_accuracy"],
            random_forest_resultats["precision"],
            random_forest_resultats["recall"],
            random_forest_resultats["f1_score"],
            random_forest_resultats["auc"]
        ],
        "SVM": [
            svm_resultats["validation_accuracy"],
            svm_resultats["precision"],
            svm_resultats["recall"],
            svm_resultats["f1_score"],
            svm_resultats["auc"]
        ],
        "Arbre de décision": [
            decision_tree_resultats["validation_accuracy"],
            decision_tree_resultats["precision"],
            decision_tree_resultats["recall"],
            decision_tree_resultats["f1_score"],
            decision_tree_resultats["auc"]
        ]
    }

    #Réccupération des performance de chaque modele
    valeurs_par_performance = {}
    for performance in performances:
        valeurs_par_performance[performance] = []
        
    for performance, i in zip(performances, range(len(performances))):
        for modele in nom_modele:
            valeurs_par_performance[performance].append(resultats[modele][i])


    x = range(len(nom_modele))
    largeur_barre = 0.15  
    couleurs = ['blue', 'orange', 'green', 'red', 'purple']

    # Sous graphique
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True) 
    axes = axes.flatten()

    # la première ligne de performances (Accuracy, Precision, Recall)
    performances_ligne1 = performances[:3]
    for i, performance in enumerate(performances_ligne1):
        decalage = largeur_barre * (i - len(performances_ligne1) / 2)
        axes[0].bar(
            [pos + decalage for pos in x], 
            valeurs_par_performance[performance], 
            largeur_barre, 
            label=performance, 
            color=couleurs[i]
        )
    axes[0].set_title("Accuracy, Precision et Recall par Modèle", fontsize=14)
    axes[0].set_ylabel("Valeur", fontsize=12)
    axes[0].legend(fontsize=10, loc='lower right')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Graphique pour la deuxième ligne de performances (F1-score, AUC)
    performances_ligne2 = performances[3:]
    for i, performance in enumerate(performances_ligne2):
        decalage = largeur_barre * (i - len(performances_ligne2) / 2)
        axes[1].bar(
            [pos + decalage for pos in x], 
            valeurs_par_performance[performance], 
            largeur_barre, 
            label=performance, 
            color=couleurs[i + len(performances_ligne1)]  # decalage dans les couleurs
        )
    axes[1].set_title("F1-score et AUC par Modèle", fontsize=14)
    axes[1].set_xlabel("Modèles", fontsize=12)
    axes[1].set_ylabel("Valeur", fontsize=12)
    axes[1].legend(fontsize=10, loc='lower right')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(nom_modele, fontsize=10)
    plt.tight_layout()
    plt.show()
    
    
    
if __name__ == "__main__":
    main()