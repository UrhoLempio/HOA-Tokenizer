from glob import glob
import math
import webdataset as wds

if __name__ == "__main__":
    path = "/Volumes/MyBook/hoa_out_speech_shards/train/"
    path = glob.glob(f"{path}/*.tar")
    dataset = wds.WebDataset(path, shardshuffle=1000)
    sample = next(iter(dataset))
    print(sample.keys())