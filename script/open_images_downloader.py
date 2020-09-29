"""
1. Download images and annotations from Open images dataset
2. Output files that fulfill yolo training
"""
import glob
import pandas as pd
import numpy as np
import wget


def get_image_list(train_annotation, train_images, classes, label):
    """
    Get the list of images with specific labels
    """
    ids = classes[classes["classes"] == label].id.values
    output = train_annotation[train_annotation.LabelName.isin(ids)]
    output["label"] = label
    output = output.merge(
        train_images[["ImageID", "Rotation"]], on="ImageID", how="left"
    )
    output = output[output.Rotation == 0]
    return output


def write_annotation(select_label, classes, imageid, location, label):
    """
    Save annotation into txt files
    """
    xmin, xmax, ymin, ymax = location
    with open("images/{}.txt".format(imageid), "a") as f_w:
        f_w.write(
            " ".join(
                [
                    select_label.index(
                        classes.loc[classes.id == label, "classes"].values[0]
                    ),
                    str((xmax + xmin) / 2),
                    str((ymin + ymax) / 2),
                    str(xmax - xmin),
                    str(ymax - ymin),
                    "\n",
                ]
            )
        )
    f_w.close()


def write_image_list(files):
    """
    Write down paths of images for training and validation
    """
    train = np.random.choice(files, size=round(len(files) * 0.8))
    with open("train.txt", "a") as f_w:
        for i in train:
            f_w.write(i + "\n")
    f_w.close()

    valid = list(set(files) - set(train))
    with open("valid.txt", "a") as f_w:
        for i in valid:
            f_w.write(i + "\n")
    f_w.close()


def main():
    """
    Download images and annotation with specific labels, and output files for yolo training.
    """
    select_label = ["Helmet", "Glasses"]
    train_annotation = pd.read_csv("oidv6-train-annotations-bbox.csv")
    train_images = pd.read_csv("train-images-boxable-with-rotation.csv")
    classes = pd.read_csv("class-descriptions-boxable.csv", names=["id", "classes"])

    select = pd.DataFrame()
    for i in select_label:
        select = select.append(
            get_image_list(train_annotation, train_images, classes, i)
        )

    select = select.reset_index(drop=True)
    select_image = train_images[train_images.ImageID.isin(select.ImageID.unique())]
    select_image = select_image.reset_index(drop=True)
    image_404 = []
    for url, imageid in select_image[["Thumbnail300KURL", "ImageID"]].values:
        try:
            wget.download(url, out="images/{}.jpg".format(imageid))
        except:
            image_404.append(imageid)

    select = select[~select.ImageID.isin(image_404)]

    # annotation files
    select = select[["ImageID", "XMin", "XMax", "YMin", "YMax", "LabelName"]].values
    for i, xmin, xmax, ymin, ymax, label in select:
        write_annotation(select_label, classes, i, (xmin, xmax, ymin, ymax), label)

    # Prepare files for training
    write_image_list(glob.glob("images/*.jpg"))


if __name__ == "__main__":
    main()
