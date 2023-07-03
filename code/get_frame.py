import glob
import os
import cv2
import math
import pandas as pd

class GetFrames(object):
  '''
  Extracts the nth frame from a video
  '''
  def __init__(self):
    self.file_path = r"C:\Users\nguye\Documents\master_thesis_project\how2sign_data\ST\raw_videos\37ZtKNf6Yd8-1-rgb_front.mp4"
    self.n_list = [0.5]
    self.file_list = []

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
    folder_path = os.path.join(os.path.dirname(directory), "picture", name + "_" + str(n))
    print(folder_path)
    # file_name = "%s.png" % folder_path
    # self.file_list.append(file_name)
    # cv2.imwrite(file_name, frame)
    # capture.release()
    print("Pic generated")

  def run(self):
    for n in self.n_list:
      self.get_nth_frame(n)
    return self.file_list

if __name__ == "__main__":
  getframe = GetFrames()
  getframe.run()
#
# class RunGetFrames(object):
#   def __init__(self, folder_path):
#     self.getframes = GetFrames()
#     self.folder_path = folder_path
#     self.file_list = []
#   def run_on_videos(self):
#     files = glob.glob(self.folder_path + "/*.mp4")
#     for file in files:
#       self.getframes.file_path = file
#       file = self.getframes.run()
#       self.file_list.extend(file)
#     file_list_df = pd.DataFrame(self.file_list, columns=["file_name"])
#     file_list_df.to_csv("/content/drive/MyDrive/asl_project/facial_landmarks_tracking/results/organized_csv/new_images.csv")
#
#     return self.file_list