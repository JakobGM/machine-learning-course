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
                new_column.append(
                    "totally_unique"
                )
            else:
                new_column.append(
                    "train_unique"
                )
            continue
        targets = train_targets[matches]
        contains_zeros = 0 in targets
        contains_ones = 1 in targets
        if contains_zeros and contains_ones:
            new_column.append("mixed_target")
        elif contains_zeros:
            new_column.append("zero_target")
        elif contains_ones:
            new_column.append("one_target")
        else:
            raise RuntimeError("Should never reach this point!")

    return (
        feature_name + "_categorical",
        pd.Categorical(
            new_column,
            categories=[
                "totally_unique",
                "train_unique",
                "mixed_target",
                "zero_target",
                "one_target",
            ],
            ordered=False,
        ),
    )
