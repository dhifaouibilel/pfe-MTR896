import requests

def get_change_details(change_id: str):
    url = f"http://localhost:8080/a/changes/{change_id}/detail"
    auth = requests.auth.HTTPBasicAuth('admin', 'password')  # à adapter
    response = requests.get(url, auth=auth)
    
    if response.status_code != 200:
        raise Exception(f"Erreur Gerrit: {response.status_code}")
    
    return response.json()
