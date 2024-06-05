import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    f1_score,
)
from ultralytics import YOLO
import os
import pandas as pd

results = np.zeros((10, 5))
labels = ["buildings", "forest", "mountain", "sea", "street"]
base_train = "./rskf_data_yolo/fold_{}/split_{}/data.yaml"
base_test = "./rskf_data_yolo/fold_{}/split_{}/test/images/"
results_labels = ["Test accuracy", "F1", "Balanced", "Precision", "Recall"]
label_true = {}

for i, label in enumerate(labels):
    type_files = os.listdir("./data/" + label)
    for file in type_files:
        label_true[file] = i

for fold in range(0, 2):
    for split in range(0, 5):
        fold_id = (fold * 5) + split

        model = YOLO("yolov8x.yaml")
        model.train(data=base_train.format(fold, split), epochs=20, imgsz=150)
        y_true = np.zeros((2815))
        y_pred = np.zeros((2815))
        count = 0

        test_path = base_test.format(fold, split)

        for file in os.listdir(test_path):
            result = model.predict(test_path + file)
            confidences = result[0].boxes.conf.cpu().numpy()
            labels = result[0].boxes.cls.cpu().numpy()
            if len(confidences) == 0:
                y_pred[count] = 5
            else:
                max_conf_idx = np.argmax(confidences)
                y_pred[count] = int(labels[max_conf_idx])
            y_true[count] = label_true[file]
            count += 1

        results[fold_id][0] = round(accuracy_score(y_true, y_pred), 3)
        results[fold_id][1] = round(f1_score(y_true, y_pred, average="weighted"), 3)
        results[fold_id][2] = round(balanced_accuracy_score(y_true, y_pred), 3)
        results[fold_id][3] = round(
            precision_score(y_true, y_pred, average="weighted"), 3
        )
        results[fold_id][4] = round(recall_score(y_true, y_pred, average="weighted"), 3)
df = pd.DataFrame(results)
df.to_csv("yolo_8_detect.csv", header=results_labels, index=False)
