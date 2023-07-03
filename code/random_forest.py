from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from typing import Tuple
import time
from typing import List
import os

class RunRandomForestModel(object):
  '''
  Runs Random Forest Model on data.
  '''
  def __init__(self):
    self.x_train_aug_file_path = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/results/cos_results/train_angles_o9.csv"
    self.y_train_aug_file_path = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/results/cos_results/train_classes_o9.csv"
    self.x_test_file_path = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/results/cos_results/test_angles_o9.csv"
    self.y_test_file_path = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/results/cos_results/test_classes_o9.csv"
    self.x_test = []
    self.y_test = []
    self.y_pred = []
    self.accuracy = 0
    self.x_train_aug = []
    self.y_train_aug = []
    self.col_names = []
    self.confusion = []
    self.report = []
    self.n_components = 0
    self.n_estimators = 200
    self.samples_leaf = 5
    self.train_time_pca = 0
    self.test_time_pca = 0
    self.train_time_rf = 0
    self.test_time_rf = 0

  def open_csv(self, file_path: str) -> pd.DataFrame:
    '''
    Read .csv file as pd.DataFrame
    '''
    df = pd.read_csv(file_path)
    return df

  def k_fold_validation(self, clf):
    '''
    Perform k-fold validation on dataset for k = 5, 7 and 10.
    '''
    scores_5 = cross_validate(clf, self.x_train_aug, self.y_train_aug, cv = 5, return_estimator=True)
    print("5 fold scores:")
    print(scores_5)
    scores_7 = cross_validate(clf, self.x_train_aug, self.y_train_aug, cv = 7, return_estimator=True)
    print("7 fold scores:")
    print(scores_7)
    scores_10 = cross_validate(clf, self.x_train_aug, self.y_train_aug, cv = 10, return_estimator=True)
    print("10 fold scores:")
    print(scores_10)

  def run_PCA(self, x_train: pd.DataFrame, x_test: pd.DataFrame):
    '''
    Runs PCA to reduce dimensionality to self.n principal components.
    '''
    pca = PCA(n_components=self.n_components, svd_solver = "full")
    self.x_train_aug = pca.fit_transform(self.x_train_aug)
    self.x_test = pca.transform(self.x_test)

  def random_forest_model(self):
    '''
    Runs random forest model on inputted data and calculates accuracy and confusion matrix using sklearn algorithms.
    '''
    clf = RandomForestClassifier(n_estimators=self.n_estimators, max_features="sqrt", random_state=0, min_samples_leaf = self.samples_leaf)
    start_train = time.time()
    if self.n_components > 0:
      pca = PCA(n_components=self.n_components, svd_solver = "full")

      start_train_pca = time.time()
      self.x_train_aug = pca.fit_transform(self.x_train_aug)
      end_train_pca = time.time()

      start_train_rf = time.time()
      clf.fit(self.x_train_aug, self.y_train_aug)
      end_train_rf = time.time()

      start_test_pca = time.time()
      self.x_test = pca.transform(self.x_test)
      end_test_pca = time.time()

      start_test_rf = time.time()
      self.y_pred = clf.predict(self.x_test)
      end_test_rf = time.time()
    else:
      clf.fit(self.x_train_aug, self.y_train_aug)
      self.y_pred = clf.predict(self.x_test)
    self.train_time_pca = end_train_pca-start_train_pca
    self.test_time_pca = end_test_pca-start_test_pca
    self.train_time_rf = end_train_rf-start_train_rf
    self.test_time_rf = end_test_rf-start_test_rf
    clf.score(self.x_train_aug, self.y_train_aug)
    self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)

  def calculate_metrics(self, class_type: str) -> Tuple[float, float, float, float]:
    '''
    Calculates the true positive rate, false positive rate, true negative rate, and false negative rate.

    Input:
    class_type = Class type to calculate metrics for
    Output:
    tpr = True positive rate
    fpr = False positive rate
    tnr = True negative rate
    fnr = False negative rate
    '''
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(self.y_pred)):
      if self.y_pred[i] == self.y_test[i]:
        if self.y_pred[i] == class_type:
          tp+=1
        elif self.y_pred[i] != class_type:
          tn+=1
      elif self.y_pred[i] != self.y_test[i]:
        if self.y_pred[i] == class_type:
          fp+=1
        elif self.y_pred[i] != class_type:
          fn+=1
    fpr = fp/(fp+tn)
    fnr = fn/(fn+tp)
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    return tpr, fpr, tnr, fnr

  def print_results(self):
    '''
    Prints the hyperparemeters of the model, the execution times, accuracy and the confusion matrix.
    '''
    print(f"Number of PCs (0 if PCA was not done): {self.n_components}")
    print(f"Model Training execution time(seconds): {self.train_time_rf}")
    print(f"Model Testing execution time(seconds): {self.test_time_rf}")
    print(f"PCA Training execution time(seconds): {self.train_time_pca}")
    print(f"PCA Testing execution time(seconds): {self.test_time_pca}")
    print(f"Number of trees: {self.n_estimators}")
    print(f"Min sample of leaves: {self.samples_leaf}" )
    print(f"Accuracy: {self.accuracy}")
    tpr_as, fpr_as, tnr_as, fnr_as = self.calculate_metrics(class_type="AS")
    print("For AS:")
    print(f"TPR: {tpr_as}, FPR: {fpr_as}, TNR: {tnr_as}, FNR: {fnr_as}")
    tpr_st, fpr_st, tnr_st, fnr_st = self.calculate_metrics(class_type="ST")
    print("For ST:")
    print(f"TPR: {tpr_st}, FPR: {fpr_st}, TNR: {tnr_st}, FNR: {fnr_st}")
    print("")

  def run(self):
    self.x_train_aug = self.open_csv(self.x_train_aug_file_path)
    self.y_train_aug = list(self.open_csv(self.y_train_aug_file_path)["class"].values)
    self.x_test = self.open_csv(self.x_test_file_path)
    self.y_test = list(self.open_csv(self.y_test_file_path)["class"].values)
    if self.n_components > 0:
      self.run_PCA(self.x_train_aug, self.x_test)
    self.random_forest_model()
    self.print_results()
    return self.train_time_rf, self.test_time_rf, self.train_time_pca, self.test_time_pca, self.y_test, self.y_pred, self.accuracy, self.x_train_aug


# parmeter types = pc, tree, leaf

class RunRFTrials():
  '''
  Run self.trials number of trials of the random forest model to get average execution times
  '''
  def __init__(self):
    self.save_folder_path = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/times/random/"
    self.trials = 10
    self.list_parameters = [4, 8, 12, 16, 20, 24, 28]
    self.parameter_type = "pc"
    self.pc = 4
    self.tree = 200
    self.leaf = 5
    self.runrandomforestmodel = RunRandomForestModel()
    self.accuracy_df = pd.DataFrame()
    self.train_times_rf_df = pd.DataFrame()
    self.test_times_rf_df = pd.DataFrame()
    self.train_times_pca_df = pd.DataFrame()
    self.test_times_pca_df = pd.DataFrame()
    self.rf_train_save_path = ""
    self.rf_test_save_path = ""
    self.pca_train_save_path = ""
    self.pca_test_save_path = ""
    self.accuracy_save_path = ""

  def run_trials(self):
    '''
    Run number of trials, calculate average times, and save results to pd.Dataframes
    '''
    for trial in range(self.trials):
      list_accuracy = []
      list_train_times_rf = []
      list_test_times_rf = []
      list_train_times_pca = []
      list_test_times_pca = []
      for parameter in self.list_parameters:
        if self.parameter_type == "tree":
          self.runrandomforestmodel.n_components = self.pc
          self.runrandomforestmodel.n_estimators = parameter
          self.runrandomforestmodel.samples_leaf = self.leaf
        if self.parameter_type == "leaf":
          self.runrandomforestmodel.n_components = self.pc
          self.runrandomforestmodel.n_estimators = self.tree
          self.runrandomforestmodel.samples_leaf = parameter
        if self.parameter_type == "pc":
          self.runrandomforestmodel.n_components = parameter
          self.runrandomforestmodel.n_estimators = self.tree
          self.runrandomforestmodel.samples_leaf = self.leaf
        train_time_rf, test_time_rf, train_time_pca, test_time_pca, y_test, y_pred, accuracy, x_train_aug = self.runrandomforestmodel.run()
        list_accuracy.append(round(accuracy, 3))
        list_train_times_rf.append(round(train_time_rf, 6)*1000) # convert to ms
        list_test_times_rf.append(round(test_time_rf, 6)*1000) # convert to ms
        list_train_times_pca.append(round(train_time_pca, 6)*1000) # convert to ms
        list_test_times_pca.append(round(test_time_pca, 6)*1000) # convert to ms
      col_name = "Trial " + str(trial+1)
      self.accuracy_df[col_name] = list_accuracy
      self.train_times_rf_df[col_name] = list_train_times_rf
      self.test_times_rf_df[col_name] = list_test_times_rf
      self.train_times_pca_df[col_name] = list_train_times_pca
      self.test_times_pca_df[col_name] = list_test_times_pca

    avg_train_trial_rf = self.train_times_rf_df.mean(axis=1)
    self.train_times_rf_df["Average"] = round(avg_train_trial_rf, 3) # in ms

    avg_test_trial_rf = self.test_times_rf_df.mean(axis=1)
    self.test_times_rf_df["Average"] = round(avg_test_trial_rf, 3)  # in ms

    avg_train_trial_pca = self.train_times_pca_df.mean(axis=1)
    self.train_times_pca_df["Average"] = round(avg_train_trial_pca, 3)  # in ms

    avg_test_trial_pca = self.test_times_pca_df.mean(axis=1)
    self.test_times_pca_df["Average"] = round(avg_test_trial_pca, 3) # in ms

    self.train_times_rf_df.insert(loc = 0, column = self.parameter_type, value = self.list_parameters)
    self.test_times_rf_df.insert(loc = 0, column = self.parameter_type, value = self.list_parameters)
    self.train_times_pca_df.insert(loc = 0, column = self.parameter_type, value = self.list_parameters)
    self.test_times_pca_df.insert(loc = 0, column = self.parameter_type, value = self.list_parameters)
    self.accuracy_df.insert(loc = 0, column = self.parameter_type, value = self.list_parameters)

  def create_file_paths(self):
    '''
    Create file_paths to save each pd.DataFrame
    '''
    self.list_parameters = [str(parameter) for parameter in self.list_parameters]
    list_parameters_string = "_".join(self.list_parameters)

    if self.parameter_type == "tree":
      rf_train_file_name = "rf_train_times" + "_pc" + str(self.pc) + "_leaf" + str(self.leaf) + "_treegrid_" + list_parameters_string + ".csv"
      self.rf_train_save_path = os.path.join(self.save_folder_path, rf_train_file_name)
      rf_test_file_name = "rf_test_times" + "_pc" + str(self.pc) + "_leaf" + str(self.leaf) + "_treegrid_" + list_parameters_string + ".csv"
      self.rf_test_save_path = os.path.join(self.save_folder_path, rf_test_file_name)

      pca_train_file_name = "pca_train_times" + "_pc" + str(self.pc) + "_leaf" + str(self.leaf) + "_treegrid_" + list_parameters_string + ".csv"
      self.pca_train_save_path = os.path.join(self.save_folder_path, pca_train_file_name)
      pca_test_file_name = "pca_test_times" + "_pc" + str(self.pc) + "_leaf" + str(self.leaf) + "_treegrid_" + list_parameters_string + ".csv"
      self.pca_test_save_path = os.path.join(self.save_folder_path, pca_test_file_name)

      accuracy_file_name = "accuracy" + "_pc" + str(self.pc) + "_leaf" + str(self.leaf) + "_treegrid_" + list_parameters_string + ".csv"
      self.accuracy_save_path = os.path.join(self.save_folder_path, accuracy_file_name)

    if self.parameter_type == "leaf":
      rf_train_file_name = "rf_train_times" + "_pc" + str(self.pc) + "_tree" + str(self.tree) + "_leafgrid_" + list_parameters_string + ".csv"
      self.rf_train_save_path = os.path.join(self.save_folder_path, rf_train_file_name)
      rf_test_file_name = "rf_test_times" + "_pc" + str(self.pc) + "_tree" + str(self.tree) + "_leafgrid_" + list_parameters_string + ".csv"
      self.rf_test_save_path = os.path.join(self.save_folder_path, rf_test_file_name)

      pca_train_file_name = "pca_train_times" + "_pc" + str(self.pc) + "_tree" + str(self.tree) + "_leafgrid_" + list_parameters_string + ".csv"
      self.pca_train_save_path = os.path.join(self.save_folder_path, pca_train_file_name)
      pca_test_file_name = "pca_test_times" + "_pc" + str(self.pc) + "_tree" + str(self.tree) + "_leafgrid_" + list_parameters_string + ".csv"
      self.pca_test_save_path = os.path.join(self.save_folder_path, pca_test_file_name)

      accuracy_file_name = "accuracy" + "_pc" + str(self.pc) + "_tree" + str(self.tree) + "_leafgrid_" + list_parameters_string + ".csv"
      self.accuracy_save_path = os.path.join(self.save_folder_path, accuracy_file_name)

    if self.parameter_type == "pc":
      rf_train_file_name = "rf_train_times" + "_tree" + str(self.tree) + "_leaf" + str(self.leaf) + "_pcgrid_" + list_parameters_string + ".csv"
      self.rf_train_save_path = os.path.join(self.save_folder_path, rf_train_file_name)
      rf_test_file_name = "rf_test_times" + "_tree" + str(self.tree) + "_leaf" + str(self.leaf) + "_pcgrid_" + list_parameters_string + ".csv"
      self.rf_test_save_path = os.path.join(self.save_folder_path, rf_test_file_name)

      pca_train_file_name = "pca_train_times" + "_tree" + str(self.tree) + "_leaf" + str(self.leaf) + "_pcgrid_" + list_parameters_string + ".csv"
      self.pca_train_save_path = os.path.join(self.save_folder_path, pca_train_file_name)
      pca_test_file_name = "pca_test_times" + "_tree" + str(self.tree) + "_leaf" + str(self.leaf) + "_pcgrid_" + list_parameters_string + ".csv"
      self.pca_test_save_path = os.path.join(self.save_folder_path, pca_test_file_name)

      accuracy_file_name = "accuracy" + "_tree" + str(self.tree) + "_leaf" + str(self.leaf) + "_pcgrid_" + list_parameters_string + ".csv"
      self.accuracy_save_path = os.path.join(self.save_folder_path, accuracy_file_name)

  def save_results(self):
    self.train_times_rf_df.to_csv(self.rf_train_save_path, index=False)
    self.test_times_rf_df.to_csv(self.rf_test_save_path, index=False)
    self.train_times_pca_df.to_csv(self.pca_train_save_path, index=False)
    self.test_times_pca_df.to_csv(self.pca_test_save_path, index=False)
    self.accuracy_df.to_csv(self.accuracy_save_path, index=False)

  def run(self):
    self.run_trials()
    self.create_file_paths()
    self.save_results()
    return self.rf_train_save_path, self.rf_test_save_path, self.pca_train_save_path, self.pca_test_save_path, self.accuracy_save_path