# Libraries to Load
import argparse
import load_and_save


def develops_aki(dataset: list, patients: list, index: int = 19) -> list:
    """
    Def of AKI:
    ● an increase in creatinine of 0.3 mg/dl (26.5 μmol/l) within 48 hours; or
    ● an increase in creatinine of 1.5 times the baseline creatinine level of
    a patient (the lowest creatinine value found in the data),
    known or presumed to have occurred within the prior 7 days.

    Note: if there are no creatine records, the patient did not suffer from AKI
    """
    table = [["id", "aki", "hours_since_admission_onset"]]
    for i, row in enumerate(dataset):
        # calculate if the patient develops aki or not
        creatinine = row[index]
        aki = False
        when = 0
        if creatinine and len(creatinine) > 2:
            n_hours = len(creatinine)
            for j in range(1, n_hours):
                if creatinine[j] and creatinine[:j]:
                    prev_48 = 0 if j < 48 else (j - 48)
                    prev_168 = 0 if j < 168 else (j - 168)
                    min_48 = min(
                        [x for x in creatinine[prev_48:j] if x is not None],
                        default=None,
                    )
                    min_168 = min(
                        [x for x in creatinine[prev_168:j] if x is not None],
                        default=None,
                    )
                    if (min_48 and (creatinine[j] >= (min_48 + 0.3))) or (
                        min_168 and (creatinine[j] >= min_168 * 1.5)
                    ):
                        aki = True
                        when = j + 1
                        break
            # save the values
            table.append([patients[i], int(aki), when])
    return table


def develops_sepsis(dataset: list, patients: list, index: int = 40) -> list:
    """
    If patient went into septic shock, it is provided in the data
    in the column 'Septic_Shock' with the last value set to 1
    """
    table = [["id", "sepsis"]]
    for i, row in enumerate(dataset):
        # calculate if the patient develops aki or not
        sepsis_onset = row[index]
        sepsis = False
        if any(sepsis_onset):
            sepsis = True
        # save the values
        table.append([patients[i], int(sepsis)])
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a labels csv from the training data."
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
    patients, features, data = load_and_save.data_transpose(fname, "train")
    if args.dataset == "aki":
        index = features.index("Creatinine")
        print("Getting AKI predictions...")
        data = develops_aki(data, patients, index)
    else:
        index = features.index("Sepsis_Onset")
        print("Getting sepsis predictions...")
        data = develops_sepsis(data, patients, index)
    save_fname = "./results/" + args.dataset + ".csv"
    print("Saving to csv...")
    load_and_save.create_csv(save_fname, data)
    print("Done. File is saved in results directory.")


if __name__ == "__main__":
    main()
