"""
CS311 Final Project: ACL Tear Diagnostic Decision Tree vs. Random Forest

Caroline Haggerty and Amy Griffin

Used PA5 as a starting point for the decision tree implementation.

"""

import argparse, os, random, sys
from typing import Any, Dict, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# Type alias for nodes in decision tree
DecisionNode = Union["DecisionBranch", "DecisionLeaf"]


class DecisionBranch:
    """Branching node in decision tree"""

    def __init__(self, attr: str, branches: Dict[Any, DecisionNode]):
        """Create branching node in decision tree

        Args:
            attr (str): Splitting attribute
            branches (Dict[Any, DecisionNode]): Children nodes for each possible value of `attr`
        """
        self.attr = attr
        self.branches = branches

    def predict(self, x: pd.Series):
        """Return predicted labeled for array-like example x"""
        pred = x[self.attr]
        subtree = self.branches.get(pred, None)
        return subtree.predict(x) 

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Test Feature", self.attr)
        for val, subtree in self.branches.items():
            print(" " * 4 * indent, self.attr, "=", val, "->", end=" ")
            subtree.display(indent + 1)


class DecisionLeaf:
    """Leaf node in decision tree"""

    def __init__(self, label):
        """Create leaf node in decision tree

        Args:
            label: Label for this node
        """
        self.label = label

    def predict(self, x):
        """Return predicted labeled for array-like example x"""
        return self.label

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Label=", self.label)

def entropy(q: float) -> float: 
    """Calculate binary entropy for probability p."""
    if q == 0 or q == 1: 
        return 0
    return -q * np.log2(q) - (1 - q) * np.log2(1 - q)

def information_gain(X: pd.DataFrame, y: pd.Series, attr: str) -> float:
    """Return the expected reduction in entropy from splitting X,y by attr"""
    
    p = sum(y == 1)  
    n = sum(y == 0) 
    pn = p + n     
    
    before_split = entropy(p / pn)
    
    remainder_atr = 0.0
    
    for at_value in X[attr].unique(): 
        y_sub = y[X[attr] == at_value] 
        pk = sum(y_sub == 1)         
        nk = sum(y_sub == 0)         
        pk_nk = pk + nk
        
        b_sub = entropy(pk / pk_nk)  
        
        remainder_atr += (pk_nk / pn) * b_sub 
    
    
    gain = before_split - remainder_atr
    return gain


def learn_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    attrs: Sequence[str],
    y_parent: pd.Series,
) -> DecisionNode:
    """Recursively learn the decision tree

    Args:
        X (pd.DataFrame): Table of examples (as DataFrame)
        y (pd.Series): array-like example labels (target values)
        attrs (Sequence[str]): Possible attributes to split examples
        y_parent (pd.Series): array-like example labels for parents (parent target values)

    Returns:
        DecisionNode: Learned decision tree node
    """
    
    if len(X) == 0:
        return DecisionLeaf(y_parent.mode()[0])
    elif len(y.unique()) == 1: 
        return DecisionLeaf(y.iloc[0]) 
    elif len(attrs) == 0: 
        return DecisionLeaf(y.mode()[0]) 

    a_max = max(attrs, key=lambda attr: information_gain(X, y, attr)) 

    branches = {} 
    grouped = X.groupby(a_max, observed=False)
    for value, group in grouped: 
        subset_X = group 
        subset_y = y[group.index] 

        remaining_attrs = []
        for a in attrs:
            if a != a_max:
                remaining_attrs.append(a)
        branches[value] = learn_decision_tree(subset_X, subset_y, remaining_attrs, y) 

    return DecisionBranch(a_max, branches)

def fit_random_forest(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_trees: int = 10, 
    max_features: int = None, 
    random_state: int = None
):
    """Train a random forest classifier with `n_trees` decision trees."""
    if random_state is not None:
        np.random.seed(random_state)
    
    forest = []
    n_samples = len(X)
    max_features = max_features or int(np.sqrt(X.shape[1])) 
    
    for _ in range(n_trees):
        bootstrap_indices = np.random.choice(X.index, size=len(X), replace=True)
        X_bootstrap = X.loc[bootstrap_indices].reset_index(drop=True)
        y_bootstrap = y.loc[bootstrap_indices].reset_index(drop=True)
    
        feature_subset = np.random.choice(X.columns, size=max_features, replace=False)

        # print(f"Training tree with {len(feature_subset)} features")
        # print(f"Features: {feature_subset}")
        # print(f"Labels: {y_bootstrap}")
        # print(f"Data: {X_bootstrap[feature_subset]}")

        tree = learn_decision_tree(X_bootstrap[feature_subset], y_bootstrap, feature_subset, y)
        forest.append((tree, feature_subset))
    
    return forest
   

def predict_random_forest(trees: list, X: pd.DataFrame) -> pd.Series:
    """Aggregate predictions from all trees in the random forest."""
    tree_predictions = []
    
    for tree, features in trees:
        predictions = predict(tree, X[features])
        tree_predictions.append(predictions)
    
    tree_predictions = pd.DataFrame(tree_predictions).T
    final_predictions = tree_predictions.mode(axis=1)[0]
    return final_predictions

def fit(X: pd.DataFrame, y: pd.Series) -> DecisionNode:
    """Return train decision tree on examples, X, with labels, y"""
    # You can change the implementation of this function, but do not modify the signature
    return learn_decision_tree(X, y, X.columns, y)


def predict(tree: DecisionNode, X: pd.DataFrame):
    """Return array-like predctions for examples, X and Decision Tree, tree"""

    # You can change the implementation of this function, but do not modify the signature

    # Invoke prediction method on every row in dataframe. `lambda` creates an anonymous function
    # with the specified arguments (in this case a row). The axis argument specifies that the function
    # should be applied to all rows.
    return X.apply(lambda row: tree.predict(row), axis=1)



def load_acl(feature_file: str, label_file: str):

    # Load the feature file
    examples = pd.read_table(
        feature_file,
        na_values=["NA", ""],  # Treat "NA" and blanks as missing values
        dtype={
            "age": "Int64",
            "sex": "category",
            "knee": "category",
            "lachman-injured": "category",
            "anterior-drawer-injured": "category",
            "lever-injured": "category",
            "physician-diagnosis": "category",
            "lachman-unaffected": "category",
            "anterior-drawer-unaffected": "category",
            "lever-unaffected": "category",
            "effusion": "category",
            "MRI": "category",
            "OR": "category",
            "ACL-lat-meniscus": "category",
            "ACL-medial": "category",
            "scope-lat-men": "category",
            "scope-med-men": "category",
            "pivot-shift-intraop-injured2": "category",
            "prev-knee-surgery": "category",
            "MOI": "category",
            "sport": "category",
            "fall": "category",
            "contact": "category",
        },
    )

    labels = pd.read_table(label_file).squeeze().rename("label")
    
    examples["age"] = pd.cut(
        examples["age"],
        bins=[0, 18, 30, 45, sys.maxsize],
        right=False,
        labels=["child", "young-adult", "mid-age", "older-adult"],
    )

    return examples, labels



# You should not need to modify anything below here
def load_examples(
    feature_file: str, label_file: str, **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load example features and labels. Additional arguments are passed to
    the pandas.read_table function.

    Args:
        feature_file (str): Delimited file of categorical features
        label_file (str): Single column binary labels. Column name will be renamed to "label".

    Returns:
        Tuple[pd.DataFrame,pd.Series]: Tuple of features and labels
    """
    return (
        pd.read_table(feature_file, dtype="category", **kwargs),
        pd.read_table(label_file, **kwargs).squeeze().rename("label"),
    )

def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary of metrics in including confusion matrix, accuracy, recall, precision and F1
    """
    return {
        "confusion": metrics.confusion_matrix(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test decision tree learner")
    parser.add_argument(
        "-p",
        "--prefix",
        default="small1",
        help="Prefix for dataset files. Allowed values: acl.",
    )
    parser.add_argument(
        "-k",
        "--k_splits",
        default=10,
        type=int,
        help="Number of splits for stratified k-fold testing",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="tree",
        help="Which model to use (single decision tree or random forest). Allowed values: tree, forest",
    )
    parser.add_argument(
        "-n", 
        "--n_trees", 
        default=10, 
        type=int, 
        help="Number of trees for random forest (if applicable)"
    )
    parser.add_argument(
        "-f", 
        "--max_features", 
        default=None, 
        type=int, 
        help="Number of features for random forest (if applicable)"
    )
    parser.add_argument(
        "-r", 
        "--random_state", 
        default=42, 
        type=int, 
        help="Random state for reproducibility"
    )

    args = parser.parse_args()

    if args.prefix == "acl":
        # We use a slightly different procedure with "adult". Instead of using a fixed split, we split
        # the data k-ways (preserving the ratio of output classes) and test each split with a Decision
        # Tree trained on the other k-1 splits.

        data_file = os.path.join(os.path.dirname(__file__), "data", "acl.data.txt")
        labels_file = os.path.join(os.path.dirname(__file__), "data", "acl.label.txt")
        data, labels = load_acl(data_file, labels_file)

        accuracy_metrics = []
        recall_metrics = []
        confusion_matrix_metrics = []
        precision_metrics = []
        f1_metrics = []

        kfold = StratifiedKFold(n_splits=args.k_splits)
        for train_index, test_index in kfold.split(data, labels):

            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            if args.model == "tree":
                # Decision tree
                tree = fit(X_train, y_train)
                y_pred = predict(tree, X_test)
                accuracy_metrics.append(metrics.accuracy_score(y_test, y_pred))
                recall_metrics.append(metrics.recall_score(y_test, y_pred))
                confusion_matrix_metrics.append(metrics.confusion_matrix(y_test, y_pred))
                precision_metrics.append(metrics.precision_score(y_test, y_pred))
                f1_metrics.append(metrics.f1_score(y_test, y_pred))
                tree.display()
            elif args.model == "forest":
                # Random forest
                forest = fit_random_forest(X_train, y_train, n_trees=args.n_trees, max_features=args.max_features, random_state=args.random_state)
                y_pred = predict_random_forest(forest, X_test)
            
                accuracy_metrics.append(metrics.accuracy_score(y_test, y_pred))
                recall_metrics.append(metrics.recall_score(y_test, y_pred))
                confusion_matrix_metrics.append(metrics.confusion_matrix(y_test, y_pred))
                precision_metrics.append(metrics.precision_score(y_test, y_pred))
                f1_metrics.append(metrics.f1_score(y_test, y_pred))

        if args.model == "tree":
            print(
                f"Mean (std) Accuracy (for k={kfold.n_splits} splits): {np.mean(accuracy_metrics)} ({np.std(accuracy_metrics)})"
                f"\nMean (std) Recall: {np.mean(recall_metrics)} ({np.std(recall_metrics)})"
                f"\nMean (std) Precision: {np.mean(precision_metrics)} ({np.std(precision_metrics)})"
                f"\nMean (std) F1: {np.mean(f1_metrics)} ({np.std(f1_metrics)})"
                f"\nConfusion Matrix:\n{np.mean(confusion_matrix_metrics, axis=0)}"
            )
        else:
            print(
                f"Mean (std) Accuracy (for k={kfold.n_splits} splits): {np.mean(accuracy_metrics)} ({np.std(accuracy_metrics)})"
                f"\nMean (std) Recall: {np.mean(recall_metrics)} ({np.std(recall_metrics)})"
                f"\nMean (std) Precision: {np.mean(precision_metrics)} ({np.std(precision_metrics)})"
                f"\nMean (std) F1: {np.mean(f1_metrics)} ({np.std(f1_metrics)})"
                f"\nMean Confusion Matrix:\n{np.mean(confusion_matrix_metrics, axis=0)}"
            )
       
    
    
