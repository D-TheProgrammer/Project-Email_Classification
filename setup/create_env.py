import os
import subprocess
import sys

#Cree un environnement virtuel
def creer_env_virutel(env_name):
    if not os.path.exists(env_name):
        print("Création de l'environnement virtuel ",env_name, "...")
        subprocess.check_call([sys.executable, "-m", "venv", env_name])
    else:
        print("L'environnement virtuel ", env_name," existe déjà.")

if __name__ == "__main__":
    env_name = "email_classifier_env_vfinal"
    creer_env_virutel(env_name)
    print("Environnement créé avec succès : ", env_name)
