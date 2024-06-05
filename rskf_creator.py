from sklearn.model_selection import RepeatedStratifiedKFold
from glob import glob
import numpy as np
import shutil

labels = ["buildings", "forest", "mountain", "sea", "street"]

X = []
y = []
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)

for i, label in enumerate(labels):
    files = glob("./data/" + label + "/*.jpg")
    for file in files:
        X.append(file)
        y.append(i)

for i, (train, test) in enumerate(rskf.split(X, y)):
    train_data = list(np.array(X)[train])
    for file in train_data:
        file_split = file.split("data/")[1]
        copy_path = (
            "./rskf_data/fold_"
            + str(int(i / 5))
            + "/split_"
            + str(i % 5)
            + "/train/"
            + file_split
        )
        shutil.copy(file, copy_path)
    test_data = list(np.array(X)[test])
    for file in test_data:
        file_split = file.split("data/")[1]
        copy_path = (
            "./rskf_data/fold_"
            + str(int(i / 5))
            + "/split_"
            + str(i % 5)
            + "/test/"
            + file_split
        )
        shutil.copy(file, copy_path)
