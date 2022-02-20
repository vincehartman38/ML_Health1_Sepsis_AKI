# Libraries to Load
import argparse
import numpy as np
import load_and_save


def sepsis_pred_matrix(data: list, labels: list, data_split: str = "train") -> list:
    table = np.zeros((len(data), 35))
    for i, matrix in enumerate(data):
        hours_admit = matrix[-2]
        np_matrix = np.array(matrix, dtype=np.float64)
        np_matrix = np_matrix[:35, :]
        # if case (developed sepsis)
        if (data_split == "train") and int(labels[i][1]):
            ow = np_matrix[:, :-6]
        else:
            # if control (did not develop sepsis)
            """if one patient is not discharged during 24 hours, the length of OW
            is set as a 24-hour time window after admission to the ICU.
            If a patient is discharged within 24 hours, we consider the entire time window
            from admission to discharge. If a control case in the training data has the
            first 24 hours of data missing, use the entire data provided per patient as the OW."""
            try:
                hour_index = hours_admit.index(24)
            except ValueError:
                hour_index = -1
            ow = np_matrix[:, :hour_index]
        mean_matrix = np.nanmean(ow, axis=1, dtype=np.float64)
        table[i] = mean_matrix.T
    # replace nan values with the median across the column
    if data_split == "train":
        col_median = np.nanmedian(table, axis=0)
        load_and_save.create_csv("./results/sepsis_col_medians.csv", [list(col_median)])
        inds = np.where(np.isnan(table))
        table[inds] = np.take(col_median, inds[1])
        table = np.concatenate(
            (np.c_[labels[:, 0]], table, np.c_[labels[:, 1]]), axis=1
        )
    else:
        col_median = load_and_save.read_csv("./results/sepsis_col_medians.csv", False)
        col_median = np.array(col_median)
        inds = np.where(np.isnan(table))
        table[inds] = np.take(col_median, inds[1])
        table = np.concatenate((np.c_[labels], table), axis=1)
    return table


def aki_pred_matrix(
    data: list, labels: list, onset_window: int = 24, data_split: str = "train"
) -> list:
    table = []
    for i, matrix in enumerate(data):
        hours_admit = matrix[-1]
        np_matrix = np.array(matrix, dtype=np.float64)
        np_matrix = np_matrix[:35, :]
        # Remove cases (developed AKI) in the prediction window
        if (
            (data_split == "test")
            or (not int(labels[i][1]))
            or (not int(labels[i][2]) < onset_window)
        ):
            ow = np_matrix[:, :onset_window]
            mean_matrix = np.nanmean(ow, axis=1, dtype=np.float64)
            if data_split == "train":
                aki = (
                    1
                    if int(labels[i][1])
                    and onset_window < int(labels[i][2]) <= (onset_window + 24)
                    else 0
                )
                table.append(
                    np.concatenate(([str(labels[i][0])], mean_matrix.T, [aki]))
                )
            else:
                table.append(
                    np.concatenate(
                        (
                            [labels[i]],
                            mean_matrix.T,
                        )
                    )
                )
    # replace nan values with the median across the column

    table = np.array(table)
    if data_split == "train":
        table_vals = table[:, 1:-1].astype(float)
        col_median = np.nanmedian(table_vals, axis=0)
        load_and_save.create_csv(
            "./results/aki_" + str(onset_window) + "_col_medians.csv",
            [list(col_median)],
        )
        inds = np.where(np.isnan(table_vals))
        table_vals[inds] = np.take(col_median, inds[1])
        table = np.hstack((np.vstack(table[:, 0]), table_vals, np.vstack(table[:, -1])))
    else:
        table_vals = table[:, 1:].astype(float)
        col_median = load_and_save.read_csv(
            "./results/aki_" + str(onset_window) + "_col_medians.csv", False
        )
        col_median = np.array(col_median)
        inds = np.where(np.isnan(table_vals))
        table_vals[inds] = np.take(col_median, inds[1])
        table = np.hstack((np.vstack(table[:, 0]), table_vals))
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a training labels matrix csv.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sepsis", "aki"],
        help="the dataset to load",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        required=True,
        choices=["train", "test"],
        help="the data split to load",
    )
    parser.add_argument(
        "--onset_window",
        type=int,
        choices=[24, 48, 72],
        help="the onset window for AKI prediction",
    )
    args = parser.parse_args()
    if not args.onset_window and (args.dataset == "aki"):
        parser.error("argument '--onset_window' is required when dataset is aki")
    return parser.parse_args()


def main():
    args = parse_args()
    fname = "./data/" + args.dataset + "/"
    print("Loading dataset...")
    pid, features, data = load_and_save.data_transpose(fname, args.data_split)
    if args.data_split == "train":
        print("Loading labels...")
        labels = load_and_save.read_csv("./results/" + args.dataset + ".csv")
        np_labels = np.array(labels)
        header = ["id"] + features[:35] + ["Label"]
    else:
        np_labels = pid
        header = ["id"] + features[:35]
    if args.dataset == "aki":
        print("Creating AKI prediction matrix...")
        data = aki_pred_matrix(data, np_labels, args.onset_window, args.data_split)
    else:
        print("Creating sepsis prediction matrix...")
        data = sepsis_pred_matrix(data, np_labels, args.data_split)
    data = np.vstack((header, data))
    save_fname = "./results/" + args.data_split + "_" + args.dataset + "_pred_matrix"
    save_fname += (
        "_" + str(args.onset_window) + ".csv" if args.dataset == "aki" else ".csv"
    )
    print("Saving to csv...")
    load_and_save.create_csv(save_fname, data)
    print("Done. File is saved in results directory.")


if __name__ == "__main__":
    main()
