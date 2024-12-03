import os
from dotenv import load_dotenv

def load_environment():
    # Charger les variables d'environnement depuis .env
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)

    # Charger GOOGLE_APPLICATION_CREDENTIALS
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if key_path:
        # Assurez-vous que la variable est un chemin absolu
        if not os.path.isabs(key_path):
            key_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', key_path))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
