import pandas as pd


def categorical_feature(args):
    feature_name, rows, train_rows, test_rows, train_targets, is_train = args
    new_column = []

    for index, row in enumerate(rows):
        matches = (train_rows == row)
        if is_train:
            matches[index] = False
        unique_in_train = not matches.any()
        if unique_in_train:
            unique_in_test = row in test_rows
            if unique_in_test:
                new_column.append(4)
            else:
                new_column.append(3)
            continue
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
