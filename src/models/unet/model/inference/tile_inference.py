import os
import threading
from queue import Queue
from time import sleep

from torch import nn

from src.models.unet.model.inference.SingleTilePredictor import SingleTilePredictor
from src.python_helpers.pipeline_config import get_dates


def __producer_file_opener(opening_queue: Queue, inference_queue: Queue):
    while not opening_queue.empty():

        # wait for inference_queue to contain less than 2 elements
        while inference_queue.qsize() >= 2:
            sleep(1)

        single_tile_predictor = opening_queue.get()

        try:
            single_tile_predictor.open_scene()
            inference_queue.put(single_tile_predictor)
        except AssertionError as e:
            print(f"[ERROR] {e}")
            print(f"[ERROR] Skipping inference for {single_tile_predictor.date}")
            print(f"[ERROR] Abort during opening scene for {single_tile_predictor.date}")
            continue


def run_inference(pipeline_config, unet: nn.Module, model_file_name: str = 'unet'):
    dates = get_dates(pipeline_config)

    # if empty, we run inference for all dates of the specified tile
    if len(dates) == 0:
        tile_name = pipeline_config["tile_name"]

        base_path = os.environ['EXTRACTED_RAW_DATA']
        folders = os.listdir(base_path)

        # filter for folders that contain the tile name
        folders = [f for f in folders if f'T{tile_name}' in f]
        s2_dates = [f.split('_')[2] for f in folders]
        print(f"[INFO] running inference for all dates of tile {tile_name}: {s2_dates}")
        print(f"[INFO] This is for {len(s2_dates)} dates")
        print("\n")

    opening_queue = Queue()
    inference_queue = Queue()

    tile_id = pipeline_config["tile_id"]
    for date in dates:
        try:
            opening_queue.put(SingleTilePredictor(pipeline_config, unet, date, tile_id, model_file_name))
        except AssertionError as e:
            print(f"[ERROR] {e}")
            print(f"[ERROR] Abort during opening scene for {date}")
            continue

    producer_thread = threading.Thread(target=__producer_file_opener, args=(opening_queue, inference_queue))
    producer_thread.start()

    while (not inference_queue.empty()) or (producer_thread.is_alive()):
        print("Waiting for inference queue to be filled...")
        single_tile_predictor = inference_queue.get(timeout=10 * 60)  # wait 10 minutes to avoid deadlocks

        # catch timeout exception
        if single_tile_predictor is None:
            print("Timeout exception")
            break

        try:
            single_tile_predictor.infer()
            single_tile_predictor.report_time()
        except AssertionError as e:
            print(f"[ERROR] {e}")
            print(f"[ERROR] Skipping inference for {single_tile_predictor.date}")
            print(f"[ERROR] Abort during inference for {single_tile_predictor.date}")
            continue

    print("Finished inference for all dates")
    producer_thread.join()
