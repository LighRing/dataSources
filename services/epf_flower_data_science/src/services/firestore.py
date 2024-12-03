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

def get_firestore_parameters():
    """
    Retrieve parameters from Firestore.
    
    Returns:
        dict: The parameters stored in the Firestore collection 'parameters'.
    """
    # Initialize Firestore client
    try:
        db = firestore.Client()

        # Retrieve the document
        collection_name = "parameters"
        document_name = "parameters"
        doc_ref = db.collection(collection_name).document(document_name)
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()
        else:
            raise ValueError(f"Document '{document_name}' not found in collection '{collection_name}'.")

    except Exception as e:
        print(f"Failed to retrieve Firestore parameters: {e}")
        raise
    
def update_firestore_parameters(updates: dict):
    """
    Update parameters in the Firestore 'parameters' document.

    Args:
        updates (dict): Dictionary of parameters to update.

    Returns:
        dict: Updated parameters from Firestore.
    """
    try:
        db = firestore.Client()

        # Collection and document
        collection_name = "parameters"
        document_name = "parameters"
        doc_ref = db.collection(collection_name).document(document_name)

        # Update the document
        doc_ref.update(updates)

        # Retrieve the updated document
        updated_doc = doc_ref.get()
        if updated_doc.exists:
            return updated_doc.to_dict()
        else:
            raise ValueError(f"Document '{document_name}' not found in collection '{collection_name}'.")
    except Exception as e:
        print(f"Failed to update Firestore parameters: {e}")
        raise

def add_firestore_parameters(new_parameters: dict):
    """
    Add or overwrite parameters in the Firestore 'parameters' document.

    Args:
        new_parameters (dict): Dictionary of parameters to add or overwrite.

    Returns:
        dict: Updated parameters from Firestore.
    """
    try:
        db = firestore.Client()

        # Collection and document
        collection_name = "parameters"
        document_name = "parameters"
        doc_ref = db.collection(collection_name).document(document_name)

        # Set (add or overwrite) the document
        doc_ref.set(new_parameters, merge=True)

        # Retrieve the updated document
        updated_doc = doc_ref.get()
        if updated_doc.exists:
            return updated_doc.to_dict()
        else:
            raise ValueError(f"Document '{document_name}' not found in collection '{collection_name}'.")
    except Exception as e:
        print(f"Failed to add Firestore parameters: {e}")
        raise

