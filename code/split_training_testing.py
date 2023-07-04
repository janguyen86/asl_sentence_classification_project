from sklearn.model_selection import train_test_split
import pandas as pd

class SplitData(object):
    """
    Splits data into training and testing.
    """
    def __init__(self):
        self.orig_csv_file = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\how2sign_data\final_dataset.csv"
        self.orig_df = pd.DataFrame()
        self.list_videos = [] #x
        self.list_class = [] #y
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.train_save_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\orig_train.csv"
        self.test_save_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\orig_test.csv"

    def open_orig_csv(self):
        self.orig_df = pd.read_csv(self.orig_csv_file, delimiter="\t")
        self.list_videos = self.orig_df["VIDEO_NAME"]
        self.list_class = self.orig_df["CLASS"]

    def save_csv(self):
        """
        Save training and testing datasets as .csv files
        :return:
        """
        self.train_df.to_csv(self.train_save_path, sep="\t", index=False)
        self.test_df.to_csv(self.test_save_path, sep="\t", index=False)

    def split_train_test(self):
        """
        Splits dataset into training and testing subset.
        :return:
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.list_videos, self.list_class, test_size=0.3,
                                                                                shuffle="True")
        self.train_df["VIDEO_NAME"] = self.x_train
        self.train_df["CLASS"] = self.y_train
        self.test_df["VIDEO_NAME"] = self.x_test
        self.test_df["CLASS"] = self.y_test

    def run(self):
        self.open_orig_csv()
        self.split_train_test()
        self.save_csv()

if __name__=="__main__":
    splitdata = SplitData()
    splitdata.run()
    print("")

