# Libraries to Load
import argparse
import statistics
import load_and_save


def gen_characteristics(dataset: list, features: list) -> list:
    data = [[] for _ in range(35)]
    for patient in dataset:
        # limit to only the 35 continuous features
        for i, row in enumerate(patient[:35]):
            data[i] += row
    distr = [["feature", "min", "max", "mean", "median", "missing", "std"]]
    for i, feature in enumerate(data):
        exc_empty = [x for x in feature if x is not None]
        # calculate the minimum
        min_v = min(exc_empty)
        # calculate the maximum
        max_v = max(exc_empty)
        # calculate the mean
        mean_v = statistics.mean(exc_empty)
        # calculcate the median
        median_v = statistics.median(exc_empty)
        # calculate the number of missing values
        missing_v = len(feature) - len(exc_empty)
        # get std
        std_v = statistics.stdev(exc_empty)
        # save the values
        distr.append([features[i], min_v, max_v, mean_v, median_v, missing_v, std_v])
    return distr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the 35 continous features distribution."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sepsis", "aki"],
        help="the dataset to load",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fname = "./data/" + args.dataset + "/"
    print("Loading dataset...")
    _, features, data = load_and_save.data_transpose(fname, "train")
    print("Getting feature distributions...")
    feature_set = gen_characteristics(data, features)
    save_fname = "./results/" + args.dataset + "_features.csv"
    print("Saving to CSV...")
    load_and_save.create_csv(save_fname, feature_set)
    print("Done. File is saved in results directory.")


if __name__ == "__main__":
    main()
