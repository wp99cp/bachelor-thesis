import base64
import json
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__, static_folder='res/imgs')
CORS(app)


def get_next_pair():
    """

    returns the url of an image pair found in "res/imgs" folder
    a pair is defined as two images which share the same coordinates x, y (file name *_x_y.png)

    :return: JSON object
    """

    file_names = os.listdir("res/imgs")
    file_names.remove('.gitignore')

    days = set()
    for file_name in file_names:
        days.add(file_name.split("_")[2].split(".")[0])

    # filter out processed images
    processed_images = os.listdir("res/masks")
    file_names = [file_name for file_name in file_names if file_name not in processed_images]

    file_name = file_names[0]
    day_of_image = file_name.split("_")[2].split(".")[0]

    days.remove(day_of_image)

    return {
        "scene_original": file_name,
        "scene_0": file_name.replace(day_of_image, days.pop()),
    }


@app.route('/next_image', methods=['GET'])
def next_image():
    return get_next_pair()


@app.route('/imgs/<path>', methods=['GET'])
def serve_images(path):
    return app.send_static_file(path)


@app.route('/update_mask/<image_name>', methods=['POST'])
def update_mask(image_name):
    # save the retrieved image to the "res/masks" folder
    # the image is saved as bytes in request.data

    data_json = json.loads(request.data)

    imgdata = base64.b64decode(data_json['image'].split(',')[1])
    filename = 'res/masks/' + image_name
    with open(filename, 'wb') as f:
        f.write(imgdata)

    return jsonify({"success": True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
