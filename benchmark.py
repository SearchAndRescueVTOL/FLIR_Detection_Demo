from ultralytics import YOLO
import os
import time
from pathlib import Path
BATCH_SIZE = 16
def benchmark(model, data_dir, out_dir):
    paths = [str(p) for p in Path(data_dir).glob("**/*")]
    start = time.time()
    avg_batch_time = 0
    counter = 0
    for i in range(0, len(paths), BATCH_SIZE):
        counter += 1
        batch = paths[i:i+BATCH_SIZE]
        results = model(batch, save = True, project = out_dir, device = 1, name = "predictions")
        batch_end = time.time()
        print(f"Processed batch {i // BATCH_SIZE + 1} in time {batch_end - start}")
        avg_batch_time += (batch_end - start)
    end = time.time()
    avg_batch_time //= counter
    total = end - start
    print("Summary:")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Num batches: {counter}")
    print(f"Total time: {total}")
    print(f"Total number of images: {counter * BATCH_SIZE}")
    print(f"Average time per batch {avg_batch_time}")
    return results

if __name__ == "__main__":
    model = YOLO("./best.pt")
    data_dir = "./data"
    out_dir = "./results/"
    results = benchmark(model, data_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)
