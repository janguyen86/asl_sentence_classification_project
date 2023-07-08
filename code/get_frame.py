import glob
import os
import cv2
import math
import pandas as pd
from typing import Tuple, List
class GetFrames(object):
  """
  Extracts the nth frame from a video where  0 < n < 1.
  """
  def __init__(self):
    self.video_file_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\how2sign_data\AS\raw_videos\3ddzkmFPEBU-1-rgb_front.mp4"
    self.n = 0.5
    self.rel_folder = "train"
    self.image_file_name = ""
    self.image_file_path = ""

  def get_nth_frame(self):
    """
    Finds the nth frame of the video and saves it as .png
    """
    directory = os.path.dirname(self.video_file_path)
    capture = cv2.VideoCapture(self.video_file_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_point = math.floor(total_frames * self.n)
    capture.set(cv2.CAP_PROP_POS_FRAMES, mid_point)
    os.chdir(directory)
    ret, frame = capture.read()
    self.image_file_name = os.path.splitext(os.path.basename(self.video_file_path))[0] + "_" + str(self.n) + ".png"
    self.image_file_path = os.path.join(os.path.dirname(directory), "images", self.rel_folder, self.image_file_name)
    cv2.imwrite(self.image_file_path, frame)
    capture.release()
    print("Image generated")

  def run(self) -> Tuple[str, str]:
    """
    Runs find nth frame and returns the new image name and its file path
    :return: image file name and file path
    """
    self.get_nth_frame()
    return self.image_file_name, self.image_file_path

class RunGetFrames(object):
  def __init__(self):
    self.getframes = GetFrames()
    self.home_directory = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\how2sign_data"
    self.csv_video_name_file_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\test.csv"
    self.save_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\orig_test_images_0.5.csv"
    self.rel_path = ""
    self.list_video_file_path = []
    self.image_file_name_list = []
    self.image_file_path_list = []
    self.list_class = []
    self.results_df = pd.DataFrame()

  def get_list_video_file_path(self):
    """
    Get list of video names, its file path and its corresponding class from .csv file.
    :return: None
    """
    video_name_df = pd.read_csv(self.csv_video_name_file_path, sep="\t")
    self.list_class = video_name_df["CLASS"]
    list_video_name = video_name_df["VIDEO_NAME"]
    self.list_video_file_path = [os.path.join(self.home_directory, class_name, "raw_videos", file_name) for class_name, file_name in zip(self.list_class, list_video_name)]
    self.rel_path = os.path.basename(self.csv_video_name_file_path).split(sep=".")[0]

  def save_file_paths(self):
    """
    Saves all the file name, path and its class of all images created to .csv file.
    :return: None
    """
    self.results_df["IMAGE_NAME"] = self.image_file_name_list
    self.results_df["IMAGE_FILE_PATH"] = self.image_file_path_list
    self.results_df["CLASS"] = self.list_class
    self.results_df.to_csv(self.save_path, sep="\t", index=False)

  def run_on_videos(self):
    """
    Get nth frame from each video and saves all file names and paths of images created to .csv file.
    :return: None
    """
    self.get_list_video_file_path()
    for video_path in self.list_video_file_path:
      self.getframes.video_file_path = video_path
      self.getframes.rel_folder = self.rel_path
      image_file_name, image_file_path = self.getframes.run()
      self.image_file_name_list.append(image_file_name)
      self.image_file_path_list.append(image_file_path)
    self.save_file_paths()

if __name__ == "__main__":
  rungetframe = RunGetFrames()
  rungetframe.csv_video_name_file_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\train.csv"
  rungetframe.save_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\orig_train_images_0.5.csv"
  rungetframe.run_on_videos()