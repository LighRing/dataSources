from google.cloud import firestore

def create_firestore_collection():
    """
    Create a Firestore collection 'parameters' with a document named 'parameters'
    containing default parameter values.
    """
    # Initialiser le client Firestore
    try:
        db = firestore.Client()

        # Collection et document
        collection_name = "parameters"
        document_name = "parameters"

        # Données par défaut
        data = {
            "n_estimators": 100,
            "criterion": "gini"
        }

        # Ajouter ou mettre à jour le document
        db.collection(collection_name).document(document_name).set(data)
        print(f"Collection '{collection_name}' and document '{document_name}' created successfully!")
    except Exception as e:
        print(f"Failed to create Firestore collection: {e}")
        raise

