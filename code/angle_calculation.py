import pandas as pd
import numpy as np

class GetAngles(object):
  """
  Calculates angles for a image based on coordinates.
  """
  def __init__(self):
    self.angle = []
    self.magnitude_list = []
    self.col_names = []
    self.origin = 8
    self.relative_coor = []
    self.x_coor = []
    self.y_coor = []
    self.x_relative_coor = []
    self.y_relative_coor = []
    self.coor = []

  # def get_relative_coordinate(self):
  #   '''
  #   Sets the selected coordinate to be the origin and calculates the other coordinates wrt to the new origin.
  #   '''
  #   origin = self.coor[self.origin]
  #   dim = np.shape(self.coor)
  #   ones = np.ones([dim[0], 1])
  #   x_origin = origin[0]
  #   y_origin = origin[1]
  #   self.x_relative_coor = [x - x_origin for x in self.x_coor]
  #   self.y_relative_coor = [y_origin - y for y in self.y_coor]
  #   self.relative_coor = np.array([[x, y] for x, y in zip(self.x_relative_coor, self.y_relative_coor)])
  #   print("")

  def get_relative_coordinate(self):
    '''
    Sets the selected coordinate to be the origin and calculates the other coordinates wrt to the new origin.
    '''
    origin = self.coor[self.origin]
    dim = np.shape(self.coor)
    ones = np.ones([dim[0], 1])
    x_origin = origin[0]
    y_origin = origin[1]
    x_origin_list = ones*x_origin
    y_origin_list = ones*y_origin
    self.x_relative_coor = self.x_coor - x_origin_list
    self.y_relative_coor = y_origin_list-self.y_coor
    self.relative_coor = np.array([[x[0], y[0]] for x, y in zip(self.x_relative_coor, self.y_relative_coor)])

  def calculate_angle(self):
    """
    Calculates the magnitude and angle using cosine of each point.
    :return:
    """
    x = self.relative_coor[:, 0]
    y = self.relative_coor[:, 1]
    squared_x = np.square(x)
    squared_y = np.square(y)
    self.mag = np.sqrt(squared_x + squared_y)
    inside = y / self.mag
    self.angle = np.arccos(inside)

  def run(self):
    self.get_relative_coordinate()
    self.calculate_angle()
    return self.angle

class RunGetAngles(object):
  def __init__(self):
    self.getangles = GetAngles()
    self.coor_csv_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\coor_orig_test_images_0.5.csv"
    self.angles_save_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\test_angle_o9_0.5.csv"
    self.classes_save_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\test_class_o9_0.5.csv"
    self.getangles = GetAngles()
    self.x_coor = []
    self.y_coor = []
    self.coor = []
    self.col_names_angles = []
    self.col_names_points = []
    self.coor_df = pd.DataFrame()
    self.angles = []
    self.classes = []
    self.angles_dict = {}
    self.angle_df = []
    self.n = 8
    self.numbers = range(0,67)

  def generate_col_names_angle(self):
    """
    Creates column names for each angle calculated
    :return:
    """
    numbers = range(0,67)
    self.col_names_angle = [f'angle{n}' for n in numbers]

  def generate_col_names_point(self):
    """
    Creates column names for each set of points.
    :return:
    """
    self.col_names_points = [f"point_{n}_{_}" for _ in ["x" , "y"] for n in self.numbers]

  def read_csv(self):
    """
    Read .csv file.
    :return: None
    """
    self.coor_df = pd.read_csv(self.coor_csv_path, sep="\t")

  def get_coor(self, row):
    """
    Reformats points data to be (x, y) for every file
    :param row: Row in csv file that contains file and its corresponding coordinates.
    :return: None
    """
    self.generate_col_names_point()
    self.x_coor = [[row[f"point_{j}_x"]] for j in range(67)]
    self.y_coor = [[row[f"point_{j}_y"]] for j in range(67)]
    self.coor = np.hstack((self.x_coor, self.y_coor))
    return self.coor

  def run_get_angles(self):
    """
    Run get angles on all the row of .csv that holds each file's coordinates.
    :return: None
    """
    for i in range(len(self.coor_df.index)):
      row = self.coor_df.iloc[i]
      self.get_coor(row)
      self.getangles.coor = self.coor
      self.getangles.x_coor = self.x_coor
      self.getangles.y_coor = self.y_coor
      angles = self.getangles.run()
      self.angles_dict[i] = np.array(angles)
    self.classes = self.coor_df["CLASS"]

  def save_to_csv(self):
    """
    Save angle results and its corresponding class to .csv files
    :return:
    """
    self.generate_col_names_angle()
    self.angle_df = pd.DataFrame.from_dict(self.angles_dict, orient="index", columns=self.col_names_angle)
    self.angle_df = self.angle_df.drop(columns=["angle" + str(self.n)])
    self.angle_df.to_csv(self.angles_save_path, index=False)
    self.classes_df = pd.DataFrame(self.classes, columns=["CLASS"])
    self.classes_df.to_csv(self.classes_save_path, index=False)

  def run(self):
    """
    Gets angles based on .csv file.
    :return: None
    """
    self.generate_col_names_point()
    self.read_csv()
    self.run_get_angles()
    self.save_to_csv()

if __name__=="__main__":
  rungetangles = RunGetAngles()
  rungetangles.coor_csv_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\coor_train_all.csv"
  rungetangles.angles_save_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\train_all_angle_o9_0.59.csv"
  rungetangles.classes_save_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\train_all_class_o9_0.59.csv"
  rungetangles.run()