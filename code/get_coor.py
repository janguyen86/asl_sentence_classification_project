from imutils import face_utils
from google.colab.patches import cv2_imshow
import dlib
import cv2
import numpy as np
from typing import List, Union, Tuple
from glob import glob
import math
import pandas as pd


class GetCoordinates(object):
    """
     Get 68 coordinates for 1 image. Must have shape_predictor_68_face_landmarks.dat file to run the dlib facial landmark tracking.
    """
    def __init__(self):
        self.image_path = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/how2sign_data/AS/picture/G3qZW-hZXaQ-5-rgb_front_0.37_rotated_0.png"
        self.model_path = r"/content/drive/MyDrive/asl_project/facial_landmarks_tracking/code/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.model_path)
        self.detector = dlib.get_frontal_face_detector()
        self.image = []
        self.gray = []
        self.coor = []
        self.face_or_nah = False
        self.flatten_coor = []

    def read_image(self):
        """
        Read an image and convert to grayscale
        :return:
        """
        self.image = cv2.imread(self.image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def run_dlib_model(self):
        """
        Runs the model on gray scale image.
        :return:
        """
        rects = self.detector(self.gray)
        for (i, rect) in enumerate(rects):
            coor = self.predictor(self.gray, rect)
            self.coor = face_utils.shape_to_np(coor)

    def draw_cooor(self):
        """
        Draw points onto the image and shows it using
        :return:
        """
        for (x, y) in self.subset:
            cv2.circle(self.image, (x, y), 2, (0, 0, 255), -1)
        cv2_imshow(self.image)

    def flatten(self):
        self.flatten_coor = [coor[i] for i in range(0, 2) for coor in self.coor]

    def run_coor(self):
        """
        Reads image and converts to grayscale. Then runs dlib model on gray scale image. A subset of points is extracted from the
        dlib output. Relative coordinates are calculated using the first coordinate in the list.
        :return:
        """
        self.read_image()
        self.run_dlib_model()
        self.flatten()
        if len(self.coor) == 0:
            print("No face detected")
            self.face_or_nah = False
        else:
            self.face_or_nah = True

    def main(self):
        self.coor = []
        self.run_coor()
        # print(self.image_path)
        return self.flatten_coor, self.face_or_nah

# if __name__== "__main__":
#   get_coor = GetCoordinates()
#   coors, face_or_nah = get_coor.main()
#   print(coors)

class RunOnFolders(object):
    """
    Runs dlib model on folders containing images.
    """
    def __init__(self):
        self.facial_detector = GetCoordinates()
        self.relative_folder_list = ["x"]
        self.home_directory = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/how2sign_data"
        self.folder_path_list = []
        self.files_dict = {}
        self.list_coor = []
        self.column_names = []
        self.list_class = []
        self.coor_dict = {}
        self.x = []
        self.y = []
        self.file_names = []
        self.false_count = 0
        self.true_count = 0
        self.rel_folder = "neutral"
        self.file_names_df = []
        self.coor_file_path = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/results/updated_raw_dlib/x_neutral_coordinates.csv"
        self.file_name_csv = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/results/file_names/train_aug_neutral_x_file_names.csv"

    def get_full_folder_paths(self):
        '''
        Generates full folder paths for every folder in relative_folder_list (list of classes)
        '''
        for folder in self.relative_folder_list:
            self.folder_path_list.append(os.path.join(self.home_directory, folder, self.rel_folder))

    def get_file_paths(self):
        '''
        Creates dictionary where key = folder name and value = list of files within folder
        '''
        file_names = list(pd.read_csv(self.file_name_csv)["file_name"].values)
        for folder, folder_path in zip(self.relative_folder_list, self.folder_path_list):
            files_png = [os.path.join(folder_path, file_name) for file_name in file_names]
            self.files_dict[folder] = files_png

    def generate_col_names(self):
        '''
        Creates column names for each set of points.
        '''
        numbers = range(0, 68)
        self.column_names = [f"point_{n}_{_}" for _ in ["x", "y"] for n in numbers]

    def create_coor_dict(self) -> Dict:
        '''
        Creates dictionary of coor and with its corresponding column name.
        '''
        coor_dict = dict(zip(self.column_names, self.list_coor))
        return coor_dict

    def run_on_images(self):
        '''
        Updates coor_dict to hold list of coor for each image as the value and
        updates list_coor to hold the list of coor for each image.
        '''
        i = 0
        for key in self.files_dict:
            files_list = self.files_dict[key]

            for file in files_list:
                self.list_coor = []
                self.facial_detector.image_path = file
                print(self.facial_detector.image_path)
                coor, face_or_nah = self.facial_detector.main()
                if face_or_nah == True:
                    self.file_names.append(os.path.basename(file))
                    self.list_class.append(key)
                    self.list_coor = coor
                    coor_dict = self.create_coor_dict()
                    self.coor_dict[i] = coor_dict
                    i = i + 1
                    self.true_count = self.true_count + 1
                else:
                    self.false_count = self.false_count + 1
                    continue

    def write_to_pd(self):
        '''
        Reformats coor data (stored in self.x) and converts to pd.DataFrame
        '''
        x = pd.DataFrame(self.coor_dict)
        self.x = x.T

    def record_dlib_data(self):
        '''
        Records folder path of each image, classa and 68 coordinates and saves it to .csv file.
        '''
        final_df = pd.DataFrame(self.file_names, columns=["file_name"])
        final_df["class"] = self.y
        final = pd.concat([final_df, self.x], axis=1)
        final.to_csv(self.coor_file_path, index=False)

    def run(self):
        self.get_full_folder_paths()
        self.get_file_paths()
        self.generate_col_names()
        self.create_coor_dict()
        self.run_on_images()
        self.write_to_pd()
        self.y = self.list_class
        self.record_dlib_data()
        print("Dlib done")
        print("Number of faces detected:" + str(self.true_count))
        print("Number of faces not detected:" + str(self.false_count))
        return self.x, self.y

if __name__ == "__main__":
    runfolders = RunOnFolders()
    x, y = runfolders.run()