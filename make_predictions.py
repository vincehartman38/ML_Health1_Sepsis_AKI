# Libraries to Load
import argparse
from xmlrpc.client import boolean
import numpy as np
import load_and_save
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def determine_hyperparameters(pred_matrix: list):
    data = [pred_matrix[i][1:] for i in range(len(pred_matrix))]
    data = np.array(data, dtype=np.float64)
    X_train = data[:, :-1]
    y_train = data[:, -1]
    # 1) Addressing the case-control imbalance problem with SMOTE
    set_smote = [False, True]
    for s in set_smote:
        print("SMOTE is set to: ", s)
        if s:
            # Set the number of neighbors hyperparameter = 5
            sm = SMOTE(random_state=42, k_neighbors=5)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        # Find best hyperparameters
        C = [0.1, 0.25, 0.5, 0.75, 1]
        penalty = ["l1", "l2"]
        n_estimators = [10, 20, 50, 80, 100]
        max_depth = [5, 10, 15, 18, 20]
        model = ["lr", "rf"]
        for m in model:
            if m == "lr":
                print("Logistic Regression Hyperparameters")
                # Normalize feature values to have zero mean and unit variance
                X_train = StandardScaler().fit_transform(X_train)
                for p in penalty:
                    for c in C:
                        clf = LogisticRegression(
                            penalty=p, C=c, solver="liblinear", random_state=42
                        ).fit(X_train, y_train)
                        f1_mean = cross_val_score(
                            clf,
                            X_train,
                            y_train,
                            cv=5,
                            scoring="f1",
                        ).mean()
                        print("C=", c, ", penalty=", p, "F1-mean=", f1_mean)
            else:
                print("\n")
                print("Random Forest Hyperparameters")
                for n in n_estimators:
                    for d in max_depth:
                        clf = RandomForestClassifier(
                            n_estimators=n, max_depth=d, random_state=42
                        ).fit(X_train, y_train)
                        f1_mean = cross_val_score(
                            clf,
                            X_train,
                            y_train,
                            cv=5,
                            scoring="f1",
                        ).mean()
                        print(
                            "n_estimators=", n, ", max_depth=", d, "F1-mean=", f1_mean
                        )
        print("\n")


def run_prediction(
    problem: str,
    train_matrix: list,
    test_matrix: list,
    set_smote: bool,
    C: float,
    penalty: str,
    n_estimators: int,
    max_depth: int,
):
    # set training data
    train_data = [train_matrix[i][1:] for i in range(len(train_matrix))]
    train_data = np.array(train_data, dtype=np.float64)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    # testing data
    features = test_matrix[0][1:]
    test_data = [test_matrix[i][1:] for i in range(1, len(test_matrix))]
    test_pids = [test_matrix[i][0] for i in range(1, len(test_matrix))]
    X_test = np.array(test_data, dtype=np.float64)
    # run SMOTE if set
    if set_smote:
        # Set the number of neighbors hyperparameter = 5
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    # build logistic regression and random forest models
    print("Building regression models...")
    X_train_scaled = StandardScaler().fit_transform(X_train)
    lr = LogisticRegression(
        penalty=penalty, C=C, solver="liblinear", random_state=42, max_iter=500
    ).fit(X_train_scaled, y_train)
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    ).fit(X_train, y_train)
    print("Saving regression and random forest train results...")
    # Linear Regression Train Results
    lr_pred = lr.predict(X_train_scaled)
    lr_accuracy = accuracy_score(y_train, lr_pred)
    lr_precision, lr_recall, _, _ = precision_recall_fscore_support(
        y_train, lr_pred, average="binary"
    )
    lr_auc = roc_auc_score(y_train, lr.predict_proba(X_train_scaled)[:, 1])
    results = [["problem", "model", "smote", "auc", "accuracy", "precision", "recall"]]
    results.append(
        [problem, "lr", str(set_smote), lr_auc, lr_accuracy, lr_precision, lr_recall]
    )
    # Random Forest Train Results
    rf_pred = rf.predict(X_train)
    rf_accuracy = accuracy_score(y_train, rf_pred)
    rf_precision, rf_recall, _, _ = precision_recall_fscore_support(
        y_train, rf_pred, average="binary"
    )
    rf_auc = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
    results.append(
        [problem, "rf", str(set_smote), rf_auc, rf_accuracy, rf_precision, rf_recall]
    )
    results_fname = "./results/" + problem + "_train_results"
    results_fname += "_smote.csv" if set_smote else ".csv"
    load_and_save.create_csv(results_fname, results)

    # list important features
    print("Saving important features...")
    a = lr.coef_[0]
    lr_feature_ids = sorted(range(len(a)), key=lambda k: a[k], reverse=True)
    lr_feature_names = [features[x] for x in lr_feature_ids][:20]
    b = rf.feature_importances_
    rf_feature_ids = sorted(range(len(b)), key=lambda k: b[k], reverse=True)
    rf_feature_names = [features[x] for x in rf_feature_ids][:20]
    ids_features = list(range(1, 21))
    feature_header = ["importance", "lr", "rf"]
    feature_table = np.hstack(
        (np.c_[ids_features], np.c_[lr_feature_names], np.c_[rf_feature_names])
    )
    feature_final = np.vstack((feature_header, feature_table))
    feat_fname = "./results/" + problem + "_features_imp"
    feat_fname += "_smote.csv" if set_smote else ".csv"
    load_and_save.create_csv(feat_fname, feature_final)

    # prediction results
    print("Saving prediction results...")
    X_test_scaled = StandardScaler().fit_transform(X_test)
    lr_final = lr.predict(X_test_scaled)
    rf_final = rf.predict(X_test)
    pred_header = ["id", "lr_result", "rf_result"]
    pred_table = np.hstack((np.c_[test_pids], np.c_[lr_final], np.c_[rf_final]))
    pred_data = np.vstack((pred_header, pred_table))
    pred_fname = "./results/" + problem + "_pred"
    pred_fname += "_smote.csv" if set_smote else ".csv"
    load_and_save.create_csv(pred_fname, pred_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a prediction csv from the training data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sepsis", "aki"],
        help="the dataset to load",
    )
    parser.add_argument(
        "--onset_window",
        type=int,
        choices=[24, 48, 72],
        help="the onset window for AKI prediction",
    )
    parser.add_argument(
        "--find_hyperparameters",
        type=str,
        required=True,
        choices=["True", "False"],
        help="run a program to find the best hyperparameters",
    )
    parser.add_argument(
        "--smote",
        type=str,
        choices=["True", "False"],
        help="determine if Synthetic Minority Oversampling Technique (SMOTE) is used",
    )
    parser.add_argument(
        "--C",
        type=float,
        choices=[0.1, 0.25, 0.5, 0.75, 1],
        help="Inverse of regularization strength for Logistic Regression model",
    )
    parser.add_argument(
        "--penalty",
        type=str,
        choices=["l1", "l2"],
        help="Norm of the penalty for Logistic Regression model",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        choices=[10, 20, 50, 80, 100],
        help="Number of trees in the forest for Random Forest model",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        choices=[5, 10, 15, 18, 20],
        help="Maximum depth of the tree for Random Forest model",
    )
    args = parser.parse_args()
    if not args.onset_window and (args.dataset == "aki"):
        parser.error("argument '--onset_window' is required when dataset is aki")
    if (args.find_hyperparameters == "False") and (
        (not args.smote)
        or (not args.C)
        or (not args.penalty)
        or (not args.n_estimators)
        or (not args.max_depth)
    ):
        parser.error(
            "smote, C, penalty, n_estimators, and max_depth are required when 'find_hyperparameters' = False"
        )
    return parser.parse_args()


def main():
    args = parse_args()
    fname = "./results/train_" + args.dataset + "_pred_matrix"
    fname += (
        ".csv" if args.dataset == "sepsis" else "_" + str(args.onset_window) + ".csv"
    )
    print("Loading train prediction matrix...")
    train_matrix = load_and_save.read_csv(fname)
    if args.find_hyperparameters == "True":
        determine_hyperparameters(train_matrix)
    else:
        fname = "./results/test_" + args.dataset + "_pred_matrix"
        fname += (
            ".csv"
            if args.dataset == "sepsis"
            else "_" + str(args.onset_window) + ".csv"
        )
        print("Loading test prediction matrix...")
        test_matrix = load_and_save.read_csv(fname, False)
        problem = args.dataset
        if args.dataset == "aki":
            ow_dict = {24: "_ow1", 48: "_ow2", 72: "_ow3"}
            problem += ow_dict[args.onset_window]
        set_smote = True if args.smote == "True" else False
        run_prediction(
            problem,
            train_matrix,
            test_matrix,
            set_smote,
            args.C,
            args.penalty,
            args.n_estimators,
            args.max_depth,
        )


if __name__ == "__main__":
    main()
