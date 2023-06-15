import copy
import os
import threading
from queue import Queue
from time import sleep

from torch import nn

from model.inference.SingleTilePredictor import SingleTilePredictor


def __producer_file_opener(opening_queue: Queue, inference_queue: Queue):
    while not opening_queue.empty():

        # wait for inference_queue to contain less than 2 elements
        while inference_queue.qsize() >= 2:
            sleep(1)

        single_tile_predictor = opening_queue.get()
        single_tile_predictor.open_date()
        inference_queue.put(single_tile_predictor)


def run_inference(pipeline_config, unet: nn.Module, model_file_name: str = 'unet'):
    # create a mask for the s2_date
    s2_dates = pipeline_config["inference"]["s2_dates"]

    # if empty, we run inference for all dates of the specified tile
    if len(s2_dates) == 0:
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

    for s2_date in s2_dates:
        opening_queue.put(SingleTilePredictor(pipeline_config, unet, s2_date, model_file_name))

    producer_thread = threading.Thread(target=__producer_file_opener, args=(opening_queue, inference_queue))
    producer_thread.start()

    while (not inference_queue.empty()) or (producer_thread.is_alive()):
        print("Waiting for inference queue to be filled...")
        single_tile_predictor = inference_queue.get(timeout=10 * 60)  # wait 10 minutes to avoid deadlocks

        # catch timeout exception
        if single_tile_predictor is None:
            print("Timeout exception")
            break

        single_tile_predictor.infer()
        single_tile_predictor.report_time()

    print("Finished inference for all dates")
    producer_thread.join()
