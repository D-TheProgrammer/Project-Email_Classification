import os
import subprocess

#Installe les dépendances dans l'environnement virtuel.
def install_dependance(env_name):
    pip_path = os.path.join(env_name, 'Scripts', 'pip.exe') if os.name == 'nt' else os.path.join(env_name, 'bin', 'pip')
    print("Installation des dépendances...")
    subprocess.check_call([pip_path, 'install', '-r', 'requirements.txt'])

if __name__ == "__main__":
    env_name = "email_classifier_env_vfinal"
    install_dependance(env_name)
    print("Dépendances installées avec succès.")
