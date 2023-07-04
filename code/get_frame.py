import glob
import os
import cv2
import math
import pandas as pd

class GetFrames(object):
  """
  Extracts the nth frame from a video where  0 < n < 1.
  """
  def __init__(self):
    self.file_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\how2sign_data\AS\raw_videos\3ddzkmFPEBU-1-rgb_front.mp4"
    self.n_list = [0.5]
    self.file_list = []
    self.rel_folder = "train"

  def get_nth_frame(self, n):
    """
    Finds the nth frame of the video and saves it as .png
    """
    directory = os.path.dirname(self.file_path)
    capture = cv2.VideoCapture(self.file_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_point = math.floor(total_frames * n)
    capture.set(cv2.CAP_PROP_POS_FRAMES, mid_point)
    os.chdir(directory)
    ret, frame = capture.read()
    name = os.path.splitext(os.path.basename(self.file_path))[0]
    folder_path = os.path.join(os.path.dirname(directory), "images", self.rel_folder, name + "_" + str(n))
    file_name = "%s.png" % folder_path
    self.file_list.append(file_name)
    cv2.imwrite(file_name, frame)
    capture.release()
    print("Pic generated")

  def run(self):
    for n in self.n_list:
      self.get_nth_frame(n)
    return self.file_list

class RunGetFrames(object):
  def __init__(self, folder_path):
    self.getframes = GetFrames()\
    self.home_directory = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\how2sign_data"
    self.csv_video_name_file_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\test.csv"
    self.folder_path = folder_path
    self.file_list = []

  def run_on_videos(self):

    return self.file_list

if __name__ == "__main__":
  getframe = GetFrames()
  getframe.run()