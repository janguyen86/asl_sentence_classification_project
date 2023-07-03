from numpy import expand_dims
# from tensorflow.python.keras.utils import load_img
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
        self.file_path = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/picture/-g0sqksgyc4-2-rgb_front.png"
        self.n = 9
        self.save_path_list = []
        self.relative_path = "test"

    def generate_save_path(self, modification_type, i):
        """
        Creates save path for each image in the form of "originalimagename_modificationtype_i.png" where i is the i-th image
        :param modification_type:
        :param i:
        :return:
        """
        directory = os.path.dirname(self.file_path)
        file_name = os.path.splitext(os.path.basename(self.file_path))[0]
        save_path = os.path.join(os.path.dirname(directory), self.relative_path, file_name + modification_type)
        extension = "_" + str(i) + ".png"
        full_save_path = os.path.join(save_path + extension)
        return full_save_path

    def rotate_image(self):
        """
        Rotate images within -90 to 90 degrees
        :return:
        """
        img = load_img(self.file_path)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(rotation_range=90)
        it = datagen.flow(samples, batch_size=1)
        for i in range(self.n):
            batch = it.next()
            image = batch[0].astype('uint8')
            modification_type = "_rotated"
            full_save_path = self.generate_save_path(modification_type, i)
            save_img(full_save_path, image)
            self.save_path_list.append(full_save_path)

    def horizontal_shift(self):
        """
        Shifts images horizontal (shifts range between -200 to 200 pixels)
        :return:
        """
        img = load_img(self.file_path)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(width_shift_range=[-200, 200])
        it = datagen.flow(samples, batch_size=1)
        for i in range(self.n):
            batch = it.next()
            image = batch[0].astype('uint8')
            modification_type = "_hshift"
            full_save_path = self.generate_save_path(modification_type, i)
            save_img(full_save_path, image)
            self.save_path_list.append(full_save_path)

    def vertical_shift(self):
        '''
        Shifts images veritcal (shifts range between +/- half the height)
        '''
        img = load_img(self.file_path)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(height_shift_range=0.5)
        it = datagen.flow(samples, batch_size=1)
        for i in range(self.n):
            batch = it.next()
            image = batch[0].astype('uint8')
            modification_type = "_vshift"
            full_save_path = self.generate_save_path(modification_type, i)
            save_img(full_save_path, image)
            self.save_path_list.append(full_save_path)

    def resize(self):
        '''
        Resize images (zoom in and out range 0.5 to 1.0)
        '''
        img = load_img(self.file_path)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
        it = datagen.flow(samples, batch_size=1)
        for i in range(self.n):
            batch = it.next()
            image = batch[0].astype('uint8')
            modification_type = "_resized"
            full_save_path = self.generate_save_path(modification_type, i)
            save_img(full_save_path, image)
            self.save_path_list.append(full_save_path)

    def run(self):
        print(self.file_path)
        self.rotate_image()
        # self.horizontal_shift()
        # self.vertical_shift()
        # self.resize()
        print("Data augmentation completed")
        return self.save_path_list

class RunOnFolders(object):
    def __init__(self):
        self.dataaug = DataAugmentation()
        self.relative_folder_list = ["test"]
        self.home_directory = r"C:\Users\nguye\Documents\master_thesis_project\how2sign_data"
        self.files_dict = {}
        self.folder_path_list = []
        self.rel_folder = "test"
        self.x_train_file_path = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/results/file_names/orig_AS_neutral.csv"
        self.file_list = []
        self.files_df = pd.DataFrame()
        self.save_path = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/results/file_names/train_aug_neutral_AS_file_names.csv"

    def get_full_folder_paths(self):
        '''
        Generates full folder paths for every folder in relative_folder_list
        '''
        for folder in self.relative_folder_list:
            self.folder_path_list.append(os.path.join(self.home_directory, folder, self.rel_folder))

    def get_file_paths(self):
        '''
        Creates dictionary where key = folder name and value = list of files within folder
        '''
        self.file_names_df = pd.read_csv(self.x_train_file_path)
        file_names = list(self.file_names_df["file_name"].values)
        for folder, folder_path in zip(self.relative_folder_list, self.folder_path_list):
            files = [os.path.join(folder_path, file_name) for file_name in file_names]
            #  + ".png"
            self.files_dict[folder] = files

    def run_on_images(self):
        '''
        Run data augmentation on each image
        '''
        for key in self.files_dict:
            files_list = self.files_dict[key]

            for file in files_list:
                self.dataaug.file_path = file
                self.dataaug.relative_path = self.rel_folder
                save_path_list = self.dataaug.run()
                self.file_list.extend(save_path_list)

    def run(self):
        self.get_full_folder_paths()
        self.get_file_paths()
        self.run_on_images()
        # self.files_df["file_name"] = self.file_list
        # self.files_df.to_csv(self.save_path)
        return self.file_list

if __name__== "__main__":
    dataaugmentation = DataAugmentation()
    dataaugmentation.file_path = r"C:\Users\nguye\Documents\master_thesis_project\how2sign_data\test\ex.png"
    dataaugmentation.run()
