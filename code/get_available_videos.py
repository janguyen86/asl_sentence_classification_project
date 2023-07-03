import pandas as pd
from zipfile import ZipFile
from typing import Dict, List
import os

class GetAvailableVideos(object):
    """
    Finds available videos based on videos in zip files. Saves file names to .csv file.
    """
    def __init__(self):
        self.csv_file_path = r"C:\Users\nguye\Documents\master_thesis_project\how2sign_data\how2sign_test.csv"
        self.zip_file_path = r"C:\Users\nguye\Documents\master_thesis_project\how2sign_data\test_raw_videos.zip"
        self.list_video_name = []
        self.df_videos = []
        self.save_path = r"C:\Users\nguye\Documents\master_thesis_project\how2sign_data\available_training_dataset_unclassified.csv"
        self.df_filenames = []

    def create_list_video_names(self, csv_file_path, reformat = True):
        """
      Creates list of videos from .csv file with all video names and its translations.
        :param csv_file_path: .csv file with video names and its translations (posted on How2Sign Website)
        :param reformat: boolean indicating if the .csv file needs to be reformatted
        :return:list of video names and
        """
        if reformat == True:
            self.df_videos = pd.read_csv(csv_file_path, delimiter="\t")
        else:
            self.df_videos = pd.read_csv(csv_file_path)
        list_video_name = list(self.df_videos["VIDEO_NAME"])
        list_video_name = ["raw_videos/" + video_name + ".mp4" for video_name in list_video_name]
        return list_video_name, self.df_videos

    def create_pd_available_videos(self, zip_file_path, list_video_name, df_videos):
        """
        Creates pd dataframe of available video names
        :param zip_file_path: file path of zipped file containing all videos
        :param list_video_name: list of video names
        :param df_videos: d
        :return: None
        """
        index_files = []
        with ZipFile(zip_file_path, "r") as zipObject:
          list_filenames = zipObject.namelist()
          i = 0
          for file_name in list_filenames:
            if file_name in list_video_name:
                i = i + 1
                index_file_name = list_video_name.index(file_name)
                index_files.append(index_file_name)
        self.df_filenames = df_videos.iloc[index_files]

    def save_to_csv(self, save_path:str, df: pd.DataFrame()):
        """
        Saves results into a .csv file
        Input:
        save_path = Save path for .csv file
        """
        df.to_csv(save_path, sep="\t", index=False)
        print("Saving completed!")

    def run(self):
        self.list_video_names, self.df_videos = self.create_list_video_names(self.csv_file_path)
        self.create_pd_available_videos(self.zip_file_path, self.list_video_names, self.df_videos)
        self.save_to_csv(self.save_path, self.df_filenames)

if __name__=="__main__":
    getavailablevideos = GetAvailableVideos()
    getavailablevideos.run()