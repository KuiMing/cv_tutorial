import pandas as pd
import numpy as np
import wget
import glob


def get_image_list(train_annotation, train_images, classes, label):
    ids = classes[classes["classes"] == label].id.values
    output = train_annotation[train_annotation.LabelName.isin(ids)]
    output["label"] = label
    output = output.merge(
        train_images[["ImageID", "Rotation"]], on="ImageID", how="left"
    )
    output = output[output.Rotation == 0]
    return output


def main():

    select_label = ["Helmet", "Glasses"]
    train_annotation = pd.read_csv("oidv6-train-annotations-bbox.csv")
    train_images = pd.read_csv("train-images-boxable-with-rotation.csv")
    classes = pd.read_csv("class-descriptions-boxable.csv", names=["id", "classes"])

    select = pd.DataFrame()
    for i in select_label:
        output = get_image_list(train_annotation, train_images, classes, i)
        select = select.append(output)

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
        with open("images/{}.txt".format(i), "a") as f_w:
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

    # Prepare files for training
    files = glob.glob("images/*.jpg")
    train = np.random.choice(files, size=round(len(files) * 0.8))
    with open("train.txt", "a") as f_w:
        for i in train:
            f_w.write(i + "\n")
    f_w.close()

    valid = list(set(files) - set(train))
    with open("valid.txt", "a") as f:
        for i in valid:
            f.write(i + "\n")
    f_w.close()


if __name__ == "__main__":
    main()