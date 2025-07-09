import json
import requests
from requests.exceptions import RequestException
import os
import utils.helpers as hpr
import shutil

from logging_config import get_logger


class OpenStackDataCollector:
    """
    A class for collecting OpenStack data from the Opendev API.
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize the data collector with a base directory.
        
        Args:
            base_dir: The base directory for data collection. If None, uses the default from helpers.
        """
        # self.DIR = f"{base_dir if base_dir is not None else hpr.DIR}/Data/"
        self.logger = get_logger()
        
        
    def get_openstack_data(self, after_date, before_date):
        """
        Perform HTTP requests to Opendev API to get the list of changes 
        related to the OpenStack repository.
        
        Args:
            before_date: Only get changes before this date in format YYYY-MM-DD.
            
        Returns:
            list: List of filenames containing saved data.
        """
        is_done = False
        size = 500
        idx = 1486
        page = 0
        # Gerrit ajoute une ligne de protection XSSI à la réponse JSON
        xssi_prefix_lines = 1
        # saved_files = []
        
        while not is_done:
            # Change O to 1916314 for more information
            params = {'O': 5000081, 'n': size, 'S': page * size}
            
            url = "https://review.opendev.org/changes/?q=repositories:{} after:{} before:{}&o={}&o={}&o={}&o={}".format(
                "openstack", after_date, before_date, "CURRENT_FILES", "MESSAGES",
                "CURRENT_COMMIT", "CURRENT_REVISION"
            )
            
            self.logger.info(f"Requesting page {page}...")
            response = requests.get(url, params=params)
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch data: {response.status_code}")
                return []

            try:
                # Supprimer le préfixe anti-XSSI (la première ligne)
                response_body = response.text.split('\n', xssi_prefix_lines)[-1]
                data = json.loads(response_body)
                return data
            except Exception as e:
                self.logger.error(f"Failed to parse response: {e}")
                return []

            self.logger.info(change_response)
            is_done = True
            continue
            
            # Skip the first line (anti-XSSI prefix) and parse the JSON
            data_per_request = change_response.text.split("\n")[1]
            data_per_request = list(json.loads(data_per_request))
            
            self.logger.info(f"Retrieved {len(data_per_request)} changes")
            
            if len(data_per_request) != 0:
                # TODO CHANGE idx to page 
                filename = f"{self.DIR}openstack_data_{idx}.json"
                self.logger.info(f"Saving to {filename}")
                
                # Convert to JSON and save to file
                data_json = json.dumps(data_per_request)
                with open(filename, "w") as jsonFile:
                    jsonFile.write(data_json)
                
                saved_files.append(filename)
                page += 1
                # TODO REMOVE LINE BELOW
                idx += 1
            else:
                is_done = True
                self.logger.info("Completed data collection - no more results found")
        
        # return saved_files
    
    # def collect_data(self, before_date="2024-06-14"):
    def collect_data(self, after_date, before_date):
        """
        Main method to collect all OpenStack data.
        
        Args:
            before_date: Only get changes before this date in format YYYY-MM-DD.
            
        Returns:
            list: List of filenames containing saved data.
        """
        self.logger.info("OpenStack Data Collection started...")
        
        start_date, start_header = hpr.generate_date("This collection started at")
        
        # Prepare directory
        self.prepare_directory()
        
        # Collect data
        changes = self.get_openstack_data(after_date, before_date)
        
        end_date, end_header = hpr.generate_date("This collection ended at")
        
        self.logger.info(start_header)
        self.logger.info(end_header)
        hpr.diff_dates(start_date, end_date)
        
        self.logger.info("OpenStack Data Collection completed\n")
        
        return changes
    
    def get_change_details(self, change_id: str):
        """Récupère les détails d'un changement depuis l'API Gerrit."""
        url = 'https://review.opendev.org/changes/{}/detail/?o={}&o={}&o={}&o={}'.format(change_id , "CURRENT_FILES", "MESSAGES", "CURRENT_COMMIT", "CURRENT_REVISION")
        try:
            response = requests.get(url)
            # print('response for change detail: ',response.text)
            response.raise_for_status()  # Lève une exception pour les statuts 4xx/5xx

            # Gerrit renvoie une ligne anti-CSRF qu’il faut retirer (si présente)
            if response.text.startswith(")]}'"):
                cleaned_json = response.text[4:]
                return response.json() if not cleaned_json else cleaned_json
            return response.json()
        except RequestException as e:
            print(f"[ERREUR] Échec de la requête vers Gerrit : {e}")
            return None
        except ValueError:
            print("[ERREUR] La réponse n’est pas un JSON valide.")
            return None
        


if __name__ == "__main__":
    collector = OpenStackDataCollector()
    collector.collect_data()