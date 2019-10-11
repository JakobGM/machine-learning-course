from collections import namedtuple
from typing import Tuple

from IPyothon.display import Markdown, display

from matplotlib import pyplot as plt

import nump as np

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

DataSets = namedtuple("DataSets", ["small", "large"])
xy = namedtuple("xy", ["X", "y"])


class DataSet:
    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> None:
        self._original_train = train
        self._original_test = test
        train = train.set_index("ID_code")
        test = test.set_index("ID_code")
        self.unlabeled_test = xy(X=test, y=None)
        self.all_train = xy(X=train.iloc[:, 2:], y=train["target"])

        assert sum(train_val_test_split) == 1.0
        self.train_val_test_split = train_val_test_split
        self._split(*self.train_val_test_split)

    def _split(
        self,
        train_split: float,
        val_split: float,
        test_split: float,
    ) -> None:
        assert train_split + val_split + test_split == 1.0
        train_X, remaining_X, train_y, remaining_y = train_test_split(
            self.all_train.X,
            self.all_train.y,
            train_size=train_split,
            test_size=val_split + test_split,
            shuffle=True,
            stratify=self.all_train.y,
            random_state=42,
        )
        if val_split == 0.0:
            self.train_split = xy(X=train_X, y=train_y)
            self.val_split = xy(X=None, y=None)
            self.test_split = xy(X=remaining_X, y=remaining_y)
            return

        val_X, test_X, val_y, test_y = train_test_split(
            remaining_X,
            remaining_y,
            train_size=val_split / (val_split + test_split),
            test_size=test_split / (val_split + test_split),
            shuffle=True,
            stratify=remaining_y,
            random_state=42,
        )
        self.train_split = xy(X=train_X, y=train_y)
        self.val_split = xy(X=val_X, y=val_y)
        self.test_split = xy(X=test_X, y=test_y)

    def resplit(self, *train_val_test_split) -> "DataSet":
        assert len(train_val_test_split) == 3
        assert sum(train_val_test_split) == 1.0
        return type(self)(
            train=self._original_train,
            test=self._original_test,
            train_val_test_split=train_val_test_split,
        )

    def evaluate(self, prediction, cutoff=0.5, plot_roc=False):
        assert len(prediction) == len(self.test_split.y)
        display(Markdown("### Evaluation Report"))

        # Convert to single numpy vector if not already
        not_numpy = not isinstance(prediction, np.ndarray)
        if not_numpy and prediction.shape[1] != 1:
            prediction = prediction["target"].to_numpy()
        elif not_numpy:
            prediction = prediction.to_numpy()

        y_true = self.test_split.y
        auc_score = roc_auc_score(y_true=y_true, y_score=prediction)
        display(Markdown(f"__AUC Score:__ {auc_score:.5f}"))

        y_pred = (prediction > cutoff)
        accuracy = accuracy_score(
            y_true=y_true,
            y_pred=y_pred,
        )
        display(Markdown(
            f"\n_Following statistics calculated with cut-off = {cutoff}_"
        ))
        display(Markdown(f"__Accuracy:__ {accuracy * 100:.2f}%"))

        tp = int(np.logical_and(y_pred == 1, y_true == 1).sum())
        tn = int(np.logical_and(y_pred == 0, y_true == 0).sum())
        fp = int(np.logical_and(y_pred == 1, y_true == 0).sum())
        fn = int(np.logical_and(y_pred == 0, y_true == 1).sum())

        sensitivity = tp / (tp + fn)
        display(Markdown(f"__Sensitivity:__ {sensitivity * 100:.2f}%"))

        specificity = tn / (tn + fp)
        display(Markdown(f"__Specificity:__ {specificity * 100:.2f}%"))

        precision = tp / (tp + fp)
        display(Markdown(f"__Precision:__ {precision * 100:.2f}%"))

        recall = tp / (tp + fn)
        display(Markdown(f"__Recall:__ {recall * 100:.2f}%"))

        confusion_matrix = pd.DataFrame(
            data=[[tp, fp], [fn, tn]],
            index=["Positive Prediction", "Negative Prediction"],
            columns=["Condition Positive", "Condition Negative"],
        )
        display(confusion_matrix)

        if plot_roc:
            fpr, tpr, threshold = roc_curve(
                y_true=y_true,
                y_score=prediction,
                pos_label=1,
            )
            plt.title("Receiver Operating Characteristic")
            plt.plot(fpr, tpr, "b", label=f"AUC = {auc_score:0.2f}")
            plt.plot([0, 1], [0, 1], "r--", label="Random Guesser")
            plt.legend(loc="lower right")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.show()

    def summary(self) -> str:
        text = ["### DataSet summary"]

        datasets = [
            ("Train", self.all_train),
            ("Train split", self.train_split),
            ("Validation split", self.val_split),
            ("Test split", self.test_split),
        ]
        for name, dataset in datasets:
            if name == "Train split":
                text.append("#### Sub-splits")
                text.append(
                    "__Train, validation, test split__ = " +
                    repr(self.train_val_test_split)
                )
            if dataset.X is None:
                continue
            train_size = len(dataset.X)
            train_positives = dataset.y.sum() / train_size
            train_negatives = 1 - train_positives
            text.append(
                f"__{name} size:__ {train_size} "
                f"__Class balance:__ {100 * train_negatives:.2f}% "
                f"/ {100 * train_positives:.1f}%"
            )

        text = "\n\n".join(text)
        return Markdown(text)


def get_datasets():
    train_large = pd.read_csv("data/train.csv")
    test_large = pd.read_csv("data/test.csv")

    test_small = pd.read_csv("data/test_small.csv")
    train_small = pd.read_csv(
        "data/train_small.csv",
        names=train_large.columns,
        header=None,
        index_col=False,
    )
    small = DataSet(train=train_small, test=test_small)
    large = DataSet(train=train_large, test=test_large)
    datasets = DataSets(small=small, large=large)
    return datasets
