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
from model_loader import load_model_1, load_model_2  # Assure-toi d’avoir ces fonctions
from model_predictor import ModelPredictor
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

# Créer une instance du prédicteur
first_model = load_model_1()
second_model = load_model_2()
predictor_m1 = ModelPredictor(first_model, model_type="m1")
predictor_m2 = ModelPredictor(second_model, model_type="m2")

class ChangeRequest(BaseModel):
    # change_id: str
    change_number: int

class ChangeResponse(BaseModel):
    message: str
    pairs_generated: bool
# , response_model=ChangeResponse
@app.post("/generate-metrics")
async def generate_metrics(request: ChangeRequest):
    try:
        change_number = int(request.change_number)
        # print(change_id)
        # Vérifier si le changement est déjà dans la base de données
        existing_change = mongo.find_change_by_number(change_number)

        if not existing_change:
            
            # Récupérer les détails du changement depuis Gerrit
            change_data = data_collector.get_change_details(change_number)
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
        else:
            print(f"change with number {change_number} is exist in DB")
            metrics = mongo.get_change_metrics_by_number(change_number)
        
        prediction_m1 = predictor_m1.predict(metrics)
        print(f"la prediction de premier model est : {prediction_m1}")
        if prediction_m1:
            pair_metrics = builder.build_pairs_metrics_for_change(change_number)
            possible_deps_numbers = builder.get_possible_deps_numbers()
            logger.info(f'pair metrics: {pair_metrics}')
            prediction_m2 = predictor_m2.predict(pair_metrics)
            predicted_pairs = predictor_m2.filter_predicted_pairs(possible_deps_numbers, prediction_m2)
            print(f"les paires predictées sont : {predicted_pairs}, avec length {len(predicted_pairs)}")
        
        else: 
            print(f"le changement avec le number = {change_number} y'a pas de dependence avec un autre changement")
            

            # return ChangeResponse(message="Métriques de nouveaux changements générées.", pairs_generated=False)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))