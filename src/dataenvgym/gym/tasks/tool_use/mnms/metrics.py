from typing import Dict, List, Literal, Optional, Tuple
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def flatten_labels(
    gt_labels: List[str], pred_labels: List[str], categories: Optional[List[str]] = None
) -> Tuple[List[int], List[int], List[str]]:
    """
    Flatten the ground truth and predicted labels into a single list of int values.
    """
    union = list(set(gt_labels + pred_labels))
    if categories:

        def gt_feature_with_categories(label):
            if label in gt_labels and label in categories:
                return categories.index(label) + 1
            return 0

        def pred_feature_with_categories(label):
            if label in pred_labels and label in categories:
                return categories.index(label) + 1
            return 0

        gt_flat = [gt_feature_with_categories(label) for label in union]
        pred_flat = [pred_feature_with_categories(label) for label in union]

    else:

        def gt_feature_simple(label):
            return 1 if label in gt_labels else 0

        def pred_feature_simple(label):
            return 1 if label in pred_labels else 0

        gt_flat = [gt_feature_simple(label) for label in union]
        pred_flat = [pred_feature_simple(label) for label in union]

    return gt_flat, pred_flat, union


class PlanAccMetrics:
    """
    Class to calculate plan accuracy for the MNM task.
    """

    def __init__(self):
        self._gt: List[List[str]] = []
        self._pred: List[List[str]] = []

    def update(self, gt_labels: List[str], pred_labels: List[str]):
        self._gt.append(gt_labels)
        self._pred.append(pred_labels)

    def compute(self) -> Dict[str, float]:
        # Binarize gt and pred, treating each gt plan as 1
        binary_gt = [1.0] * len(self._gt)
        binary_pred = []
        for gt, pred in zip(self._gt, self._pred):
            binary_pred += [1.0] if pred == gt else [0.0]
        acc = accuracy_score(binary_gt, binary_pred)
        return {"accuracy": round(float(100 * acc), 2)}


class PRFMetrics:
    """
    Class to calculate the precision, recall, f1 metrics for the MNM task.
    """

    def __init__(
        self,
        categories: Optional[List[str]],
        average: Optional[Literal["micro", "macro", "binary"]],
    ):
        self.categories = categories
        self.average = average
        self.labels = list(range(1, len(categories) + 1)) if categories else None
        self._gt_flat: List[int] = []
        self._pred_flat: List[int] = []
        self._union: List[str] = []

    def update(self, gt_labels: List[str], pred_labels: List[str]):
        gt_flat, pred_flat, union = flatten_labels(
            gt_labels, pred_labels, categories=self.categories
        )
        self._gt_flat.extend(gt_flat)
        self._pred_flat.extend(pred_flat)
        self._union.extend(union)

    def get_appropriate_labels(self) -> Optional[List[int]]:
        # Macro average will break if there are categories that are not present in the union
        # This happens if we are only evaluating on a subset of the categories, for example.
        if self.average == "macro" and self.labels:
            union_labels = set(self._gt_flat) | set(self._pred_flat)
            return [label for label in self.labels if label in union_labels]
        return self.labels

    def compute(self) -> Dict[str, float]:
        appropriate_labels = self.get_appropriate_labels()
        precision, recall, f1, _ = precision_recall_fscore_support(
            self._gt_flat, self._pred_flat, labels=appropriate_labels, average=self.average  # type: ignore
        )
        return {
            "precision": round(float(100 * precision), 2),
            "recall": round(float(100 * recall), 2),
            "f1": round(float(100 * f1), 2),
        }
