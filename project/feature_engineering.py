from multiprocessing import Pool

import pandas as pd

import datasets as ds


def categorical_feature(args):
    feature_name, rows, train_rows, test_rows, train_targets, is_train = args
    new_column = []

    for index, row in enumerate(rows):
        matches = (train_rows == row)
        if is_train:
            matches[index] = False
        unique_in_train = not matches.any()
        if unique_in_train:
            test_matches = (test_rows == row)
            if not is_train:
                test_matches[index] = False
            unique_in_test = not test_matches.any()

            if unique_in_test:
                new_column.append(4)
            else:
                new_column.append(3)
            continue

        if is_train:
            matches[index] = True

        targets = train_targets[matches]
        contains_zeros = 0 in targets
        contains_ones = 1 in targets
        if contains_zeros and contains_ones:
            new_column.append(2)
        elif contains_zeros:
            new_column.append(0)
        elif contains_ones:
            new_column.append(1)
        else:
            raise RuntimeError("Should never reach this point!")

    return (
        feature_name + "_categorical",
        pd.Categorical(
            new_column,
            categories=[0, 1, 2, 3, 4],
            ordered=False,
        ),
    )


def categorical_uniqueness(dataset):
    train = dataset.all_train
    test = dataset.unlabeled_test

    train_iterable = zip(
        train.X.columns,
        train.X.to_numpy().T,
        train.X.to_numpy().T,
        test.X.to_numpy().T,
        [train.y.to_numpy()] * 200,
        [True] * 200,
    )
    pool = Pool()
    new_train_columns = {}
    for result in pool.imap(categorical_feature, train_iterable):
        print(f"Finished processing: {result[0]} (TRAIN)", end="\r")
        new_train_columns[result[0]] = result[1]

    print()
    test_iterable = zip(
        train.X.columns,
        test.X.to_numpy().T,
        train.X.to_numpy().T,
        test.X.to_numpy().T,
        [train.y.to_numpy()] * 200,
        [False] * 200,
    )
    pool = Pool()
    new_test_columns = {}
    for result in pool.imap(categorical_feature, test_iterable):
        print(f"Finished processing: {result[0]} (TEST)", end="\r")
        new_test_columns[result[0]] = result[1]

    new_train = train.X.copy()
    for column_name, values in new_train_columns.items():
        new_train[column_name] = pd.Categorical(
            values,
            categories=[0, 1, 2, 3, 4],
            ordered=False,
        )
    new_train["target"] = train.y.copy()

    new_test = test.X.copy()
    for column_name, values in new_test_columns.items():
        new_test[column_name] = pd.Categorical(
            values,
            categories=[0, 1, 2, 3, 4],
            ordered=False,
        )

    engineered_dataset = ds.DataSet(train=new_train, test=new_test)
    return engineered_dataset
