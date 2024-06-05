import os

labels = ["buildings/", "forest/", "mountain/", "sea/", "street/"]
encoding = "{} 0.5 0.5 1 1"

for fold in range(0, 2):
    for split in range(0, 5):
        for typ in ["train/", "test/"]:
            outer_save_path = "./rskf_data_yolo/fold_{}/split_{}/{}/labels/".format(
                fold, split, typ
            )
            outer_read_path = "./rskf_data/fold_{}/split_{}/{}/".format(
                fold, split, typ
            )
            for i, label in enumerate(labels):
                files = os.listdir(outer_read_path + label)
                for file in files:
                    with open(outer_save_path + file[:-4] + ".txt", "w") as write_file:
                        write_file.write(encoding.format(i))
