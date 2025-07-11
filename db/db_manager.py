from pymongo import MongoClient, DESCENDING, ASCENDING
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
from typing import Union
from transform_data import OpenStackDataTransformer
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logging_config import get_logger


logger = get_logger()
data_transformer = OpenStackDataTransformer()

class MongoManager:
    def __init__(self, db_name: str = "ml_plugin"):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]

        # Créer les 3 collections (MongoDB les créera au premier insert si non existantes)
        self.changes = self.db["changes"]
        self.metrics = self.db["metrics"]
        self.pairs = self.db["pairs"]
        self.dependencies = self.db["dependencies"]
        self.dependencies2 = self.db["dependencies2"]
        self.ensure_indexes()
        
    def find_change_by_id(self, change_id: str):
        return self.db.changes.find_one({"_id": change_id})
    
    def find_change_by_number(self, number: int):
        return self.db.changes.find_one({"number": number})
    
    def get_last_change(self):
        return self.db.changes.find_one(sort=[("created", DESCENDING)])


    def ensure_indexes(self):
        self.db.changes.create_index(
            [("project", ASCENDING), ("created", ASCENDING)],
            name="project_created_index"
        )
        self.db.changes.create_index(
            [("owner_account_id", ASCENDING), ("created", ASCENDING)],
            name="owner_created_index"
        )
        self.db.changes.create_index([("created", 1), ("number", 1)])
        
        self.db.metrics.create_index([("created", 1)])

        # self.db.changes.create_index("number")
        # self.db.changes.create_index("change_id")
        # self.db.changes.create_index("status")
        print("✅ Indexes ensured.")
        
    def simplify_change_document(self, doc):
        # print('test db simplify_change_document: ', doc.get('revisions')[0].get('files'))
        messages = data_transformer.extract_messages(doc.get("messages", []))
        revisions = data_transformer.retrieve_revisions(doc.get('revisions'))
        commit_message = data_transformer.retrieve_commit_message(doc)
        changed_files = data_transformer.filter_files_attr(revisions)
        
        return {
        "id": doc.get("id"),
        "project": doc.get("project"),
        "branch": doc.get("branch"),
        "change_id": doc.get("change_id"),
        "subject": doc.get("subject"),
        "status": doc.get("status"),
        "created": doc.get("created"),
        "updated": doc.get("updated"),
        "insertions": doc.get("insertions"),
        "deletions": doc.get("deletions"),
        "reviewers": doc.get("reviewers", {}).get("REVIEWER", []),
        "messages": messages,
        "number": doc.get("_number") or doc.get("number"),
        "revisions": revisions,
        "current_revision": doc.get("current_revision"),  # adaptation
        "owner_account_id": doc.get("owner", {}).get("_account_id"),
        "owner_name": doc.get("owner", {}).get("name"),
        "owner_username": doc.get("owner", {}).get("username"),
        "is_owner_bot": int("SERVICE_USER" in doc.get("owner", {}).get("tags", [])) if doc.get("owner") else 0,
        "total_comment_count": doc.get("total_comment_count", 0),
        "commit_message": commit_message,  
        "changed_files": changed_files
            # doc.get('revisions')[0].get('files')
            #   # ou à extraire des `revisions` si disponible
    }
        
    def insert_data(self, collection_name: str, data: Dict[str, Any]) -> str:
        """Insère un document dans la collection spécifiée."""
        if collection_name == 'changes':
            data = self.simplify_change_document(data)
        collection = getattr(self, collection_name, None)
        if collection is None:
            raise ValueError(f"Collection '{collection_name}' not found.")
        result = collection.insert_one(data)
        return data
        # return str(result.inserted_id)
    
    def save_to_db(self, collection_name, df: pd.DataFrame) -> int:
        """Insère un DataFrame dans la collection `changes`."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Le paramètre doit être un DataFrame Pandas")

        # Nettoyage : on enlève les colonnes non sérialisables comme 'NaT'
        # df = df.fillna(None)

        records = df.to_dict(orient="records")
        collection = getattr(self, collection_name, None)
        if collection is None:
            raise ValueError(f"Collection '{collection_name}' not found.")
        
        # print("⚠️ This is test")
        if not records:
            logger.info("⚠️ DataFrame vide : aucun document inséré.")
            return 0
        try:
        # Create a new connection inside the method if needed
            result = collection.insert_many(records)
            logger.info(f"✅ {len(result.inserted_ids)} documents insérés dans {collection_name}.")
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'insertion: {str(e)}")
            return 0
        #result = self.changes.insert_many(records)
        #print(f"✅ {len(result.inserted_ids)} documents insérés dans `changes`.")
        #return len(result.inserted_ids)

    def read_all(self, collection_name: str) -> pd.DataFrame:
        """Lit tous les documents d’une collection."""
        logger.info(f"Reading the collection: {collection_name}")
        collection = getattr(self, collection_name, None)
        if collection is None:
            raise ValueError(f"Collection '{collection_name}' not found.")
        # Fetch all documents
        cursor = collection.find()

        # Convert to DataFrame
        df = pd.DataFrame(list(cursor))

        # Optionally drop the MongoDB-specific '_id' field
        df.drop(columns=['_id'], inplace=True, errors='ignore')
        
        if collection_name == "changes":
            df = df.drop_duplicates(subset=["number"])

            df = df.sort_values(by="created", ascending=True).reset_index(drop=True)

            df['created'] = df['created'].map(lambda x: x[:-10])

            df["created"] = pd.to_datetime(df["created"])

            df.loc[df['project'].str.startswith('openstack/'), 'project'] = df['project'].map(lambda x: x[10:])

        logger.info(f"Collection {collection_name} loaded successfully!")
        return df
    # def find_change_by_id(self, change_id: str)-> pd.DataFrame:
    #     """Retourne un changement spécifique à partir de son change_id.
    #     """
    #     print('chercher un changement spécifique à partir de son change_id')
    #     return self.db['changes'].find_one({"change_id": change_id})
        
    def read_filtered_changes(self, num_docs: int = 0) -> pd.DataFrame:
        """
        Lit uniquement les attributs nécessaires depuis la collection 'changes' pour accélérer le traitement.
        """
        logger.info("Reading selected fields from collection: changes")

        collection = getattr(self, 'changes', None)
        if collection is None:
            raise ValueError("Collection 'changes' not found.")

        # Définir les champs à récupérer (projection MongoDB)
        fields_to_select = {
            'number': 1,
            'project': 1,
            'created': 1,
            'owner_account_id': 1,
            'changed_files': 1,
            'status': 1,
            # 'branch': 1,0
            'insertions': 1,
            'deletions': 1,
            'subject': 1,
            'commit_message': 1,
            # 'change_id':1,0
            'reviewers': 1,
            '_id': 0  # exclure le champ _id
        }

        # cursor = collection.find({}, fields_to_select)
        # cursor = collection.find({}, {
        #     'number': 1,
        #     'project': 1,
        #     'created': 1,
        #     'owner_account_id': 1,
        #     'changed_files': 1,
        #     'status': 1,'_id': 0})
        # kif na5o date moch mawjoud f premiers 50000 project_age=0
        # normallement 3adi puisque a7na la nouvelle chg ykoun created mte3o mn 50000 w normallement plus nouveau
        if num_docs and num_docs > 0:
            logger.info(f"Loading last {num_docs} documents from changes (sorted DESC)...")
            cursor = collection.find({}, fields_to_select).sort("created", 1).limit(num_docs)
        else:
            logger.info("Loading first 50000 documents from changes (sorted ASC)...")
            cursor = collection.find({}, fields_to_select).sort("created", 1).limit(50000)
        data = []
        for doc in cursor:
            data.append(doc)
        df = pd.DataFrame(data)
        # df = pd.DataFrame(list(cursor))

        df = df.drop_duplicates(subset=["number"])
        df = df.sort_values(by="created", ascending=True).reset_index(drop=True)

        df['created'] = df['created'].map(lambda x: x[:-10] if isinstance(x, str) and len(x) > 10 else x)
        df["created"] = pd.to_datetime(df["created"], errors='coerce')

        # df.loc[df['project'].str.startswith('openstack/'), 'project'] = df['project'].map(lambda x: x[10:])

        logger.info("Filtered collection 'changes' loaded successfully.")
        return df
    
    def get_metrics_by_change_numbers(self, change_numbers: list) -> pd.DataFrame:
        """
        Récupère les métriques pour une liste donnée de numéros de changement.

        Args:
            change_numbers (list): Liste des numéros de changement (Gerrit change numbers).

        Returns:
            pd.DataFrame: DataFrame contenant les métriques correspondantes.
        """
        collection = getattr(self, 'metrics', None)
        logger.info(f"Loading metrics for all changes number from metrics collection")
        query = {"number": {"$in": change_numbers}}
        cursor = collection.find(query, {"_id":0})
        logger.info("Filtered collection 'metrics' loaded successfully.")

        return pd.DataFrame(list(cursor))
    
    def get_change_metrics_by_number(self, number: int) -> Optional[Dict]:
        """
        Récupère un document depuis la collection 'metrics' en fonction de 'number' de changement.
        """
    
        try:
            collection = self.db['metrics']
            metrics = collection.find_one({'number': number}, {"_id":0})
            
            if metrics:
                logger.info(f"✅ Document trouvé pour number={number}")
            else:
                logger.warning(f"❌ Aucun document trouvé pour number={number}")
            return metrics
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du document avec number={number}: {e}")
            return None
        
    def get_change_by_number(self, number: int) -> Optional[Dict]:
        """
        Récupère un document depuis la collection 'changes' en fonction de son 'number'.
        """
    
        try:
            collection = self.db['changes']
            change = collection.find_one({'number': number}, {
        "number": 1,
        "project": 1,
        "owner_account_id": 1,
        "created": 1,  # ✅ Assure-toi que cette ligne est bien présente
        "reviewers": 1,
        "changed_files": 1,
        "insertions": 1,
        "deletions": 1,
        "commit_message": 1,
        "subject":1,
        "_id": 0
    })  # exclude MongoDB internal _id
            if isinstance(change["created"], str) and len(change["created"]) > 10:
                change["created"] = change["created"][:-10]

            change["created"] = pd.to_datetime(change["created"], errors='coerce')
            if change:
                logger.info(f"✅ Document trouvé pour number={number}")
            else:
                logger.warning(f"❌ Aucun document trouvé pour number={number}")
            return change
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du document avec number={number}: {e}")
            return None
        
        
    def add_data_test(self):
        change_doc = {
            "id": "uuid-1",
            "project": "openstack",
            "branch": "main",
            "change_id": "Iabc123",
            "owner": "bilel",
            "subject": "Fix memory leak",
            "status": "MERGED",
            "created": datetime.now(),
            "updated": datetime.now(),
            "added_lines": 200,
            "deleted_lines": 100,
            "revisions": {"rev1": "commit_hash1"},
            "_number": 12345,
            "current_revision": "rev1"
        }
        self.insert_data('changes',change_doc)
        logger.info("✅ Document inséré dans `changes`.")
        self.close()
        
    def get_last_change_date(self):
        last_change = self.db.changes.find_one(sort=[("created", DESCENDING)])
        # print('last change: ', last_change)
        return last_change["created"] if last_change else None
        
    def vide_db(self, collection_name):
        
        self.db.drop_collection(collection_name)
        logger.info(f"✅  collection {collection_name} deleted ")
        self.close()

    def close(self):
        self.client.close()



mongo = MongoManager()

# mongo.vide_db('changes')
# print(mongo.read_filtered_changes().head(10))