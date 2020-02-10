import os
import numpy as np
import pandas as pd
import tarfile
import shutil
from PIL import Image, ImageDraw


def CUB_loader(path=''):
    """unpacked CUB dataset loader
     source for dataset http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
      Parameters
      ----------
      path_folder : str
          The location of unpaked folder CUB_200_2011
          (if None then location is work dirrectory)
      Returns
      -------
      pandas DataFrame
          a dataframe 1 column PIL image, second str name of class
      """
    if path == '':
        path = os.path.join(os.getcwd(), "CUB_200")

    train_files = [s.rstrip() for s in open(os.path.join(path, "lists", "train.txt")).readlines()]
    test_files = [s.rstrip() for s in open(os.path.join(path, "lists", "test.txt")).readlines()]
    files = [s.strip() for s in open(os.path.join(path, "lists", "files.txt")).readlines()]
    labels = pd.read_csv(os.path.join(path, "attributes", "labels.txt"), header=None, sep=" ")
    attributes = pd.read_csv(os.path.join(path, "attributes", "attributes.txt"), header=None)
    certainties = pd.read_csv(os.path.join(path, "attributes", "certainties.txt"), header=None, sep=" ")
    images = pd.read_csv(os.path.join(path, "attributes", "images.txt"), header=None, sep=" ")

    attributes[0] = attributes[0].str.split(" ").apply(lambda x: " ".join(x[1:]))

    train_frame = pd.DataFrame([], columns=['image', 'class', *list(attributes[0])], )
    test_frame = pd.DataFrame([], columns=['image', 'class', *list(attributes[0])], )

    count = 0
    for file in files:
        path_file = os.path.join(path, "images", file).rstrip()
        image = Image.open(path_file)

        example = pd.DataFrame([], columns=['image', 'class', *list(attributes[0])], )
        example['image'] = [image]
        example['class'] = [file[4:file.find("/")]]

        img_id = images[images[1] == file[file.find("/") + 1:]][0].real[0]
        cur_labels = labels[labels[0] == img_id]

        for i in range(attributes.shape[0]):
            attr_name = attributes[0][i]
            cur_attr = cur_labels[cur_labels[1] == i]
            if cur_attr.empty:
                example[attr_name] = -1
            else:
                example[attr_name] = round(cur_labels[3].mean())

        if file in train_files:
            train_frame = pd.concat([train_frame, example], axis=0, ignore_index=True)
        elif file in test_files:
            test_frame = pd.concat([test_frame, example], axis=0, ignore_index=True)

        count += 1
        print(count * 1.0 / len(files))
        # if count * 1.0 / len(files) > 0.05:
        #     break
    return train_frame, test_frame


if __name__ == "__main__":
    train, test = CUB_loader("C:\\WinterSchool\\ZeroShotEval\\data\\CUB-200")
