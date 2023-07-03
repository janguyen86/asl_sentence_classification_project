from zipfile import ZipFile
from get_available_videos import GetAvailableVideos
import pandas as pd
import os

class UnzipFiles(object):
  """
  Unzips videos based on video names contained in .csv file
  """
  def __init__(self):
    self.csv_file_path = r"C:\Users\nguye\Documents\master_thesis_project\how2sign_data\available_training_dataset_classified.csv"
    self.list_video_name = []
    self.list_video_rel_path = []
    self.df_videos = []
    self.list_class = []
    self.zip_file_path = r"C:\Users\nguye\Documents\master_thesis_project\how2sign_data\test_raw_videos.zip"
    self.data_directory = r"C:\Users\nguye\Documents\master_thesis_project\how2sign_data"
    self.getavailablevideos = GetAvailableVideos()

  def get_video_info(self):
    """
    Creates lists of the video name, video relative path file, and corresponding class.
    :return: None
    """
    self.df_videos = pd.read_csv(self.csv_file_path, delimiter="\t")
    self.list_video_name = self.df_videos["VIDEO_NAME"]
    self.list_video_rel_path = ["raw_videos/" + video_name + ".mp4" for video_name in self.list_video_name]
    self.list_class = self.df_videos["CLASS"]

  def unzip_videos(self):
    """
    Unzips videos from .zip file based on video names in .csv file and saves it to the corresponding folder based on class
    :return: None
    """
    with ZipFile(self.zip_file_path, "r") as zipObject:
      list_zipped_files = zipObject.namelist()
      for video_name, video_rel_path, class_name in zip(self.list_video_name, self.list_video_rel_path, self.list_class):
        if video_rel_path in list_zipped_files:
          if class_name == "AS" or class_name == "ST":
            print(class_name)
            save_path = os.path.join(self.data_directory, class_name)
            # zipObject.extract(video_rel_path, save_path)
          else:
            print(f"x: {video_name}")
            continue
        else:
          print(f"Not in zipped file: {video_rel_path}")

  def run(self):
    self.get_video_info()
    self.unzip_videos()

if __name__=="__main__":
  unzipfiles = UnzipFiles()
  unzipfiles.run()