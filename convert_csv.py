import argparse
import numpy as np
import os
import pandas as pd
from multiprocessing import Manager
import concurrent.futures
import queue
from tqdm import tqdm
import json


def worker(task_queue, args, pbar):
    """Worker function for parallel processing of rows."""
    while True:
        try:
            index = task_queue.get(timeout=1)
        except queue.Empty:
            break

        row = csv.iloc[index]
        video_path = row["video path"]
        annotation_path = os.path.join(args.dir_path, row["annotation path"])

        json_file = json.load(open(os.path.join(annotation_path, "caption.json")))
        prompt = json_file["ShotImmersion"]

        new_csv.loc[len(new_csv)] = [video_path, annotation_path, prompt]
        task_queue.task_done()
        pbar.update(1)


def args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, default="raw.csv", help="Path to the input CSV file"
    )
    parser.add_argument(
        "--dir_path", type=str, default="./SpatialVID", help="Path to the SpatialVID directory"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of parallel workers"
    )
    return parser.parse_args()


def main():
    args = args_parser()
    global csv
    csv = pd.read_csv(args.csv_path)
    global new_csv
    new_csv = pd.DataFrame(columns=["video", "annotation_path", "prompt"])

    manager = Manager()
    task_queue = manager.Queue()
    for index in range(len(csv)):
        task_queue.put(index)

    with tqdm(total=len(csv), desc="Finished tasks") as pbar:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            futures = []
            for _ in range(args.num_workers):
                futures.append(executor.submit(worker, task_queue, args, pbar))
            for future in concurrent.futures.as_completed(futures):
                future.result()

    new_csv.to_csv("metadata.csv", index=False)


if __name__ == "__main__":
    main()
