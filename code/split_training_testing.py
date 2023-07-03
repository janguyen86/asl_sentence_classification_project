from sklearn.model_selection import train_test_split
import pandas as pd

class SplitData(object):
    """
    Splits data into training and testing.
    """
    def __init__(self):
        self.orig_csv_file = r"C:\Users\nguye\Documents\master_thesis_project\how2sign_data\available_training_dataset_classified.csv"
        self.x = []
        self.y = []
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    def open_orig_csv(self, file_path) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        return df

    def open_files(self):
        """
        Opens .csv files of the facial points and classes and reformats it.
        :return:
        """
        self.x = self.open_csv(self.x_file_path)
        self.y = list(self.open_csv(self.y_file_path)["class"].values)

    def save_csv(self, pd, save_path):
        """
        Save dataframe as .csv.
        :param pd: pandas dataframe to be saved
        :param save_path: file_path = file path to save .csv to
        :return:
        """
        pd.to_csv(save_path)

    def split_train_test(self):
        """
        Splits dataset into training and testing subset.
        :return:
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3,
                                                                                shuffle="True")

