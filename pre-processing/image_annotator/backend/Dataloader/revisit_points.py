import json
import os

from Dataloader import Dataloader
from Dataloader.Dataloader import MASKS_DIR


def load_center_points(date: str, dataloader: Dataloader):
    path = MASKS_DIR + '/' + date
    file_name = path + "/revisions.geojson"

    # check if the file exists
    if not os.path.exists(file_name):
        return []

    # Open the GeoJSON file, a JSON file with a special format
    with open(file_name) as f:
        data = json.load(f)

        coords = []
        for feature in data['features']:
            coords.append(feature['geometry']['coordinates'])

        return coords
