# Libraries to Load
import os
import csv
from pathlib import Path

# Make data and results directories if they do not exist
Path("./data").mkdir(parents=True, exist_ok=True)
Path("./results").mkdir(parents=True, exist_ok=True)

# load dataset
def data_transpose(datapath: str, data_split: str) -> list:
    f_path = datapath + data_split + "/"
    files = os.listdir(f_path)
    data = []
    patients = []
    first_line = True
    for infile in files:
        with open(f_path + infile, newline="") as f:
            if first_line:
                features = next(f).rstrip().split(",")
            else:
                next(f)
            reader = csv.reader(f)
            dataset = [
                [float(i) if i != "" else None for i in list(i)]
                for i in zip(*list(reader))
            ]
        data.append(dataset)
        patient_id, _ = os.path.splitext(infile)
        patients.append(patient_id)
    return patients, features, data


def read_csv(f_path: str, skip_header: bool = True) -> list:
    with open(f_path, newline="") as f:
        if skip_header:
            next(f)
        reader = csv.reader(f)
        dataset = list(reader)
    return dataset


# Save CSV file
def create_csv(filepath: str, data: list):
    with open(filepath, "w", newline="") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerows(data)
