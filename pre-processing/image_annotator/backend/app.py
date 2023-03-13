import json
import os
import threading
import time
import zipfile
from io import BytesIO
from queue import Queue

import matplotlib

from Dataloader.revisit_points import load_center_points

matplotlib.use('Agg')

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from Dataloader.Dataloader import Dataloader, img_src, MASKS_DIR

app = Flask(__name__, static_folder=img_src)
CORS(app)

dataloader = Dataloader()
queued_refs = Queue()


def load_next():
    queued_refs.put(dataloader.generate_next())


@app.route('/next_image', methods=['GET'])
def next_image():
    thread = None
    if queued_refs.qsize() < 5:
        # run the dataloader in a separate thread
        thread = threading.Thread(target=load_next)
        thread.start()

    if queued_refs.empty() and thread is not None:

        # spin until the dataloader has finished
        while queued_refs.empty():
            pass

    assert not queued_refs.empty(), "The queue is empty, but the dataloader is not running"

    ref = queued_refs.get()
    print("\n\n**********\n\nServing ref: " + ref + " window=" +
          str(dataloader.refs[ref]["window"]) + "\n\n**********\n\n")

    scenes = dataloader.refs[ref]["scenes"]
    scenes = {
        f"scene_original" if i == 0 else f"scene_{i - 1}": scenes[i]
        for i in range(len(scenes))
    }

    return {"ref": ref, "scenes": scenes}


@app.route('/imgs/<path>', methods=['GET'])
def serve_images(path):
    return app.send_static_file(path)


@app.route('/update_mask/<ref>', methods=['POST'])
def update_mask(ref):
    dataloader.mark_scene_as_done(ref)

    # save the retrieved image to the "res/masks" folder
    # the image is saved as bytes in request.data
    data_json = json.loads(request.data)
    imgdata = data_json['image']
    dataloader.update_mask(ref, imgdata)

    return jsonify({"success": True})


@app.route('/download/masks', methods=['GET'])
def download_masks():
    date = time.strftime("%Y%m%d-%H%M%S")
    fileName = "annotations_{}.zip".format(date)
    memory_file = BytesIO()

    file_path = MASKS_DIR

    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(file_path):
            for file in files:
                zipf.write(os.path.join(root, file))
    memory_file.seek(0)

    return send_file(memory_file, as_attachment=True, download_name=fileName)


if __name__ == '__main__':
    points_of_interest = load_center_points(dataloader.current_date, dataloader)
    dataloader.set_points_of_interest(points_of_interest)

    queued_refs.put(dataloader.generate_next())
    app.run(host='0.0.0.0', port=5000)
