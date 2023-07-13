from numpy import expand_dims
from keras.utils import load_img
from keras.utils import img_to_array
from keras.utils import save_img
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

class DataAugmentation(object):
    """
    Performs 4 different types of modification to an image and 9 variations of each modification.
    """
    def __init__(self):
        self.file_path = r"'C:\\Users\\nguye\\Documents\\GitHub\\asl_sentence_classification_project\\how2sign_data\\AS\\images\\train\\g05yGRoZE10-8-rgb_front_0.5.png'"
        self.n = 9
        self.save_path_list = []
        self.image_name_list = []
        self.relative_path = "train"

    def generate_save_path(self, augmentation_type, i):
        """
        Creates save path for each image in the form of "originalimagename_modificationtype_i.png" where i is the i-th image
        :param augmentation_type: Type of augmentation done
        :param i: ith image
        :return: None
        """
        directory = os.path.dirname(self.file_path)
        file_name = os.path.splitext(os.path.basename(self.file_path))[0]
        image_name = file_name + augmentation_type + "_" + str(i) + ".png"
        save_path = os.path.join(os.path.dirname(directory), self.relative_path, "augmented_images")
        if os.path.exists(save_path) == False:
            os.mkdir(save_path)
        full_save_path = os.path.join(save_path, image_name)
        return full_save_path, image_name

    def rotate_image(self):
        """
        Rotate images within -90 to 90 degrees
        :return: None
        """
        img = load_img(self.file_path)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(rotation_range=90)
        it = datagen.flow(samples, batch_size=1)
        for i in range(self.n):
            batch = it.next()
            image = batch[0].astype('uint8')
            augmentation_type = "_rotated"
            full_save_path, image_name = self.generate_save_path(augmentation_type, i)
            save_img(full_save_path, image)
            self.save_path_list.append(full_save_path)
            self.image_name_list.append(image_name)

    def horizontal_shift(self):
        """
        Shifts images horizontal (shifts range between -200 to 200 pixels)
        :return: None
        """
        img = load_img(self.file_path)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(width_shift_range=[-200, 200])
        it = datagen.flow(samples, batch_size=1)
        for i in range(self.n):
            batch = it.next()
            image = batch[0].astype('uint8')
            augmentation_type = "_hshift"
            full_save_path, image_name = self.generate_save_path(augmentation_type, i)
            save_img(full_save_path, image)
            self.save_path_list.append(full_save_path)
            self.image_name_list.append(image_name)

    def vertical_shift(self):
        """
        Shifts images vertical (shifts range between +/- half the height)
        :return: None
        """
        img = load_img(self.file_path)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(height_shift_range=0.5)
        it = datagen.flow(samples, batch_size=1)
        for i in range(self.n):
            batch = it.next()
            image = batch[0].astype('uint8')
            augmentation_type = "_vshift"
            full_save_path, image_name = self.generate_save_path(augmentation_type, i)
            save_img(full_save_path, image)
            self.save_path_list.append(full_save_path)
            self.image_name_list.append(image_name)

    def resize(self):
        """
        Resize images (zoom in and out range 0.5 to 1.0)
        :return: None
        """
        img = load_img(self.file_path)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
        it = datagen.flow(samples, batch_size=1)
        for i in range(self.n):
            batch = it.next()
            image = batch[0].astype('uint8')
            augmentation_type = "_resized"
            full_save_path, image_name = self.generate_save_path(augmentation_type, i)
            save_img(full_save_path, image)
            self.save_path_list.append(full_save_path)
            self.image_name_list.append(image_name)

    def run(self):
        print(self.file_path)
        self.rotate_image()
        self.horizontal_shift()
        self.vertical_shift()
        self.resize()
        print("Data augmentation completed")
        return self.save_path_list, self.image_name_list

class RunDataAugmentation(object):
    """
    Run data augmentation on a series of images from .csv file and saves results to new .csv.
    """
    def __init__(self):
        self.dataaug = DataAugmentation()
        self.csv_train = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\orig_train_images_0.5.csv"
        self.csv_save_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\train_aug.csv"
        self.list_orig_file_path = []
        self.list_orig_class = []
        self.list_aug_file_name = []
        self.list_aug_file_path = []
        self.list_aug_class = []
        self.train_aug_df = pd.DataFrame()

    def open_csv(self):
        """
        Opens image name, image path, and class of original images from .csv.
        :return: None
        """
        train_df = pd.read_csv(self.csv_train, sep="\t")
        self.list_orig_file_path = train_df["IMAGE_FILE_PATH"]
        self.list_orig_class = train_df["CLASS"]

    def save_csv(self):
        """
        Save image name, image path, and class of all new images created to .csv.
        :return: None
        """
        self.train_aug_df["IMAGE_NAME"] = self.list_aug_file_name
        self.train_aug_df["IMAGE_FILE_PATH"] = self.list_aug_file_path
        self.train_aug_df["CLASS"] = self.list_aug_class
        self.train_aug_df.to_csv(self.csv_save_path, sep="\t", index=False)

    def run_aug_images(self):
        """
        Run data augmentation for all images in .csv file and saves new image info to .csv.
        :return: None
        """
        self.open_csv()
        for file_path, class_type in zip(self.list_orig_file_path, self.list_orig_class):
            self.dataaug.file_path = file_path
            save_path_list, image_name_list = self.dataaug.run()
            list_new_class = [class_type for i in range(36)]
            self.list_aug_class.extend(list_new_class)
        self.list_aug_file_path = save_path_list
        self.list_aug_file_name = image_name_list
        self.save_csv()

if __name__== "__main__":
    rundataaugmentation = RunDataAugmentation()
    rundataaugmentation.run_aug_images()