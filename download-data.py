import json
import requests
import os
import utils.helpers as hpr
import shutil


def get_openstack_data(dir):
    """Perform http requests to Opendev API,
    to get the list of changes related to the Openstack repository
    """
    is_done = False
    size = 500
    page = 0

    while (not is_done):
        # change O to 1916314 for more informations 
        params = {'O': 1916314, 'n': size, 'S': page * size}

        url = "https://review.opendev.org/changes/?q=repositories:{} after:{}&o={}&o={}&o={}&o={}".format(
            "openstack", "2011-01-01", "CURRENT_FILES", "MESSAGES",
            "CURRENT_COMMIT", "CURRENT_REVISION")

        change_response = requests.get(url, params=params)

        data_per_request = change_response.text.split("\n")[1]

        data_per_request = list(json.loads(data_per_request))

        print("Length %d" % len(data_per_request))

        if len(data_per_request) != 0:

            print("Page %s" % page)

            data_per_request = json.dumps(data_per_request)

            jsonFile = open("{}openstack_data_{}.json".format(dir, page), "w")

            jsonFile.write(data_per_request)

            jsonFile.close()

            page += 1
        else:
            is_done = not is_done


if __name__ == "__main__":

    print("Script openstack-data-collection.py started...")

    start_date, start_header = hpr.generate_date("This script started at")

    DIR = "%sData/" % hpr.DIR

    if os.path.exists(DIR):
        shutil.rmtree(path=DIR)
    os.makedirs(DIR, exist_ok=True)

    get_openstack_data(DIR)

    end_date, end_header = hpr.generate_date("This script ended at")

    print(start_header)

    print(end_header)

    hpr.diff_dates(start_date, end_date)

    print("Script openstack-data-collection.py ended\n")
