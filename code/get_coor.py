from imutils import face_utils
import dlib
import cv2
import os
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
        self.image_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\how2sign_data\AS\images\test\-g45vqccdzI-1-rgb_front_0.5.png"
        self.model_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\code\shape_predictor_68_face_landmarks.dat"
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
        '''
        Draw points onto the image.
        '''
        for i in range(len(self.coor)):
            (x, y) = self.coor[i]
            if i == 8:
                cv2.circle(self.image, (x, y), 2, (0, 0, 255), -1)
            else:
                cv2.circle(self.image, (x, y), 2, (0, 0, 0), -1)

        cv2.imshow("coor image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        # self.draw_cooor()
        # print(self.image_path)
        return self.flatten_coor, self.face_or_nah

# if __name__== "__main__":
#   get_coor = GetCoordinates()
#   coors, face_or_nah = get_coor.main()
#   print(coors)

class RunGetCoordinates(object):
    def __init__(self):
        self.csv_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\orig_test_images_0.5.csv"
        self.getcoordinates = GetCoordinates()
        self.list_image_file_path = []
        self.list_class = []
        self.list_image_name = []
        self.list_face_detected_file_path = []
        self.list_face_detected_class = []
        self.list_image_name = []
        self.column_names = []
        self.false_count = 0
        self.true_count = 0
        self.total_count = 0
        self.final_df = pd.DataFrame()

    def get_image_file_paths(self):
        file_df = pd.read_csv(self.csv_path, sep="\t")
        self.list_image_file_path = file_df["IMAGE_FILE_PATH"]
        self.list_class = file_df["CLASS"]
        self.list_image_name = file_df["IMAGE_NAME"]

    def generate_col_names(self):
        """
        Creates column names for each set of points.
        :return:
        """
        numbers = range(0, 68)
        self.column_names = [f"point_{n}_{_}" for _ in ["x", "y"] for n in numbers]
        beginning_column_list = ["IMAGE_NAME", "CLASS"]
        beginning_column_list.extend(self.column_names)
        self.column_names = beginning_column_list

    def set_up_final_df(self):
        self.generate_col_names()
        self.final_df = pd.DataFrame(columns=self.column_names)

    def run_dlib_on_images(self):
        for image_file_path, class_type, image_name in zip(self.list_image_file_path, self.list_class, self.list_image_name):
            self.total_count = self.total_count + 1
            self.getcoordinates.image_path = image_file_path
            coors, face_or_nah = self.getcoordinates.main()
            if face_or_nah == True:
                image_row = [image_name, class_type]
                image_row.extend(coors)
                self.final_df.loc[len(self.final_df.index)] = image_row
                self.true_count = self.true_count + 1
            else:
                self.false_count = self.false_count + 1
            print(self.total_count)

    def save_results_csv(self):
        directory = os.path.dirname(self.csv_path)
        new_file_name = "coor_" + os.path.basename(self.csv_path)
        save_path = os.path.join(directory, new_file_name)
        self.final_df.to_csv(save_path, sep="\t", index=False)

    def run(self):
        self.get_image_file_paths()
        self.set_up_final_df()
        self.run_dlib_on_images()
        self.save_results_csv()
        print("Number of faces detected:" + str(self.true_count))
        print("Number of faces not detected:" + str(self.false_count))

if __name__=="__main__":
    rungetcoordinates = RunGetCoordinates()
    rungetcoordinates.csv_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\train_all.csv"
    rungetcoordinates.run()