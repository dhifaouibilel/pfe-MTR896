import json
import requests
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
        self.DIR = f"{base_dir if base_dir is not None else hpr.DIR}/Data/"
        self.logger = get_logger()
        
    def prepare_directory(self):
        """
        Prepare the data directory by removing existing files and creating a fresh directory.
        """
        if os.path.exists(self.DIR):
            shutil.rmtree(path=self.DIR)
        os.makedirs(self.DIR, exist_ok=True)
        print(f"Data directory prepared: {self.DIR}")
        
    def get_openstack_data(self, before_date="2024-06-14"):
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
        page = 0
        saved_files = []
        
        while not is_done:
            # Change O to 1916314 for more information
            params = {'O': 1916314, 'n': size, 'S': page * size}
            
            url = "https://review.opendev.org/changes/?q=repositories:{} before:{}&o={}&o={}&o={}&o={}".format(
                "openstack", before_date, "CURRENT_FILES", "MESSAGES",
                "CURRENT_COMMIT", "CURRENT_REVISION"
            )
            
            self.logger.info(f"Requesting page {page}...")
            change_response = requests.get(url, params=params)
            
            # Skip the first line (anti-XSSI prefix) and parse the JSON
            data_per_request = change_response.text.split("\n")[1]
            data_per_request = list(json.loads(data_per_request))
            
            self.logger.info(f"Retrieved {len(data_per_request)} changes")
            
            if len(data_per_request) != 0:
                filename = f"{self.DIR}openstack_data_{page}.json"
                self.logger.info(f"Saving to {filename}")
                
                # Convert to JSON and save to file
                data_json = json.dumps(data_per_request)
                with open(filename, "w") as jsonFile:
                    jsonFile.write(data_json)
                
                saved_files.append(filename)
                page += 1
            else:
                is_done = True
                self.logger.info("Completed data collection - no more results found")
        
        return saved_files
    
    def collect_data(self, before_date="2024-06-14"):
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
        saved_files = self.get_openstack_data(before_date)
        
        end_date, end_header = hpr.generate_date("This collection ended at")
        
        self.logger.info(start_header)
        self.logger.info(end_header)
        hpr.diff_dates(start_date, end_date)
        
        self.logger.info("OpenStack Data Collection completed\n")
        
        return saved_files


if __name__ == "__main__":
    collector = OpenStackDataCollector()
    collector.collect_data()