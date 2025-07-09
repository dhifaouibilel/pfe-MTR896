def extract_features(change_data: dict):
    """
    Extrait les features pour la prédiction à partir des données Gerrit
    """
    num_files = len(change_data.get("revisions", {}))
    subject_length = len(change_data.get("subject", ""))
    
    # Exemple simple – à adapter à ton modèle
    return [num_files, subject_length]
