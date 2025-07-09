#!/usr/bin/env python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.db_manager import MongoManager
from calculate_metrics import MetricService
from calculate_metrics_m2 import PairMetricsService
import json
from logging_config import get_logger
from db.db_manager import MongoManager
from collect_data import OpenStackDataCollector
# from gerrit.gerrit_client import get_change_details, get_changes_between_dates
# from features.extractor import extract_features_for_new_changes
# from metrics_model2 import generate_pair_metrics  # À adapter selon ton fichier
from datetime import datetime

app = FastAPI()
logger = get_logger()
data_collector = OpenStackDataCollector()
mongo = MongoManager()
metric_service = MetricService()
builder = PairMetricsService()

class ChangeRequest(BaseModel):
    change_id: str

class ChangeResponse(BaseModel):
    message: str
    pairs_generated: bool
# , response_model=ChangeResponse
@app.post("/generate-metrics")
async def generate_metrics(request: ChangeRequest):
    try:
        change_id = request.change_id
        # print(change_id)
        # Vérifier si le changement est déjà dans la base de données
        existing_change = mongo.find_change_by_id(change_id)

        if existing_change:
            # Générer les métriques de paires (Model 2)
            logger.info('Générer les métriques de paires')
            # generate_pair_metrics(request.change_id)
            # return ChangeResponse(message="Métriques de paires générées.", pairs_generated=True)

        else:
            # Récupérer les détails du changement depuis Gerrit
            change_data = data_collector.get_change_details(str(change_id))
            # print('change detail: ', change_data)
            if not change_data:
                raise HTTPException(status_code=404, detail="Changement introuvable dans Gerrit")
            change_data = json.loads(change_data)
            # print('change detail: ',  change_data)
            
            
            #appliquer filter new changes sauvegarder et retourner cette change for calculate metrics
            saved_change = mongo.insert_data('changes', change_data)
            
            # created_str = change_data.get('created')
            # print('change created at : ', created_str)
            # if not created_str:
            #     raise HTTPException(status_code=400, detail="Date de création manquante dans les données Gerrit")

            # change_created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
            # print('change created at: ', change_created)
            # # Dernière date connue dans la base de données
            # last_known_change = mongo.get_last_change_date()
            # print('last date in BD: ', last_known_change)
            # # last_date = last_known_change['created'] if last_known_change else datetime.now().isoformat()
            # last_date = last_known_change if last_known_change else datetime.now().isoformat()
            # Récupérer tous les changements entre la date du changement et la dernière date connue
            # new_changes = data_collector.collect_data(change_created.isoformat(), last_date)
            # logger.info('changes: ', new_changes)
            
            print(saved_change['number'])
            # Calculer les métriques pour le nouvel changement (Model 1)
            metrics = metric_service.generate_metrics_for_change(saved_change)
            pair_metrics = builder.build_pairs_metrics_for_change(saved_change['number'])
            
            logger.info(f'pair metrics: {pair_metrics}')
            

            # return ChangeResponse(message="Métriques de nouveaux changements générées.", pairs_generated=False)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))