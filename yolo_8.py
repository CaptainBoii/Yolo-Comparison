import numpy as np
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    f1_score,
)
import pandas as pd

labels = ["buildings", "forest", "mountain", "sea", "street"]
base_train = "./rskf_data/fold_{}/split_{}/"
base_test = "./rskf_data/fold_{}/split_{}/test/"
results = np.zeros((10, 5))
results_labels = ["Test accuracy", "F1", "Balanced", "Precision", "Recall"]

for fold in range(0, 2):
    for split in range(0, 5):
        fold_id = (fold * 5) + split
        model = YOLO("yolov8x-cls.yaml")

        model.train(data=base_train.format(fold, split), epochs=20)

        y_true = np.zeros((2815))
        y_pred = np.zeros((2815))

        counter = 0

        for score, label in enumerate(labels):
            results_ = model.predict(base_test.format(fold, split) + label)

            for result in results_:
                y_pred[counter] = result.probs.top1
                y_true[counter] = score
                counter += 1
        results[fold_id][0] = round(accuracy_score(y_true, y_pred), 3)
        results[fold_id][1] = round(f1_score(y_true, y_pred, average="weighted"), 3)
        results[fold_id][2] = round(balanced_accuracy_score(y_true, y_pred), 3)
        results[fold_id][3] = round(
            precision_score(y_true, y_pred, average="weighted"), 3
        )
        results[fold_id][4] = round(recall_score(y_true, y_pred, average="weighted"), 3)

df = pd.DataFrame(results)
df.to_csv("yolo_8.csv", header=results_labels, index=False)
