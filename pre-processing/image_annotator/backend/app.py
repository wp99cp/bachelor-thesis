import base64
import json
import os

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')

from flask import Flask, jsonify, request
from flask_cors import CORS

from Dataloader.Dataloader import Dataloader
from Dataloader.Dataloader import TMP_DIR

img_src = f"{TMP_DIR}/res/imgs"
if not os.path.exists(img_src):
    os.makedirs(img_src)

app = Flask(__name__, static_folder=img_src)
CORS(app)

dataloader = Dataloader()


@app.route('/next_image', methods=['GET'])
def next_image():
    scenes = dataloader.get_scenes_next_scenes()

    scene_names = []

    for i, scene in enumerate(scenes):
        random_hash = np.random.randint(0, 1000000000)
        img = Image.fromarray(np.uint8(scene * 255).transpose(1, 2, 0))
        img.save(f'{img_src}/scene_{random_hash}.png')
        scene_names.append(f'scene_{random_hash}.png')

    print("\n\n=======================================\n\n")

    return {
        f"scene_original" if i == 0 else f"scene_{i - 1}": scene_names[i]
        for i in range(len(scene_names))
    }


@app.route('/imgs/<path>', methods=['GET'])
def serve_images(path):
    return app.send_static_file(path)


@app.route('/update_mask/<image_name>', methods=['POST'])
def update_mask(image_name):
    dataloader.mark_scene_as_done()

    # save the retrieved image to the "res/masks" folder
    # the image is saved as bytes in request.data
    data_json = json.loads(request.data)
    imgdata = base64.b64decode(data_json['image'].split(',')[1])
    dataloader.update_mask(imgdata)

    return jsonify({"success": True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
