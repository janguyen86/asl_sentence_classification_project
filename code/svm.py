from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple
import time
from typing import List
import os

class RunSVM(object):
  '''
  Runs SVM on data.
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
    self.n_components = 20
    self.kernel = "rbf"
    self.degree = 3
    self.gamma = "auto"
    self.train_time_pca = 0
    self.test_time_pca = 0
    self.train_time_svm = 0
    self.test_time_svm = 0

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

  def SVM(self):
    '''
    Runs SVM on inputted data and calculates accuracy and confusion matrix using sklearn algorithms.
    '''
    clf = make_pipeline(StandardScaler(), SVC(degree = self.degree,  kernel = self.kernel, gamma = self.gamma))
    if self.n_components > 0:
      pca = PCA(n_components=self.n_components, svd_solver = "full")

      start_train_pca = time.time()
      self.x_train_aug = pca.fit_transform(self.x_train_aug)
      end_train_pca = time.time()

      start_train_svm = time.time()
      clf.fit(self.x_train_aug, self.y_train_aug)
      end_train_svm = time.time()

      start_test_pca = time.time()
      self.x_test = pca.transform(self.x_test)
      end_test_pca = time.time()

      start_test_svm = time.time()
      self.y_pred = list(clf.predict(self.x_test))
      end_test_svm = time.time()

    else:
      clf.fit(self.x_train_aug, self.y_train_aug)
      self.y_pred = list(clf.predict(self.x_test))
    self.train_time_pca = end_train_pca-start_train_pca
    self.test_time_pca = end_test_pca-start_test_pca
    self.train_time_svm = end_train_svm-start_train_svm
    self.test_time_svm = end_test_svm-start_test_svm
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
    print(f"Number of PCs (0 if PCA was not done): {self.n_components}")
    print(f"Model Training execution time(seconds): {self.train_time_svm}")
    print(f"Model Testing execution time(seconds): {self.test_time_svm}")
    print(f"PCA Training execution time(seconds): {self.train_time_pca}")
    print(f"PCA Testing execution time(seconds): {self.test_time_pca}")
    print(f"Kernel: {self.kernel}")
    print(f"Degree (ignore if kernel not poly): {self.degree}")
    print(f"gamma: {self.gamma}")
    print(f"Accuracy: {self.accuracy}")
    tpr_as, fpr_as, tnr_as, fnr_as = self.calculate_metrics(class_type="AS")
    print("For AS:")
    print(f"TPR: {tpr_as}, FPR: {fpr_as}, TNR: {tnr_as}, FNR: {fnr_as}")
    tpr_st, fpr_st, tnr_st, fnr_st = self.calculate_metrics(class_type="ST")
    print("For ST:")
    print(f"TPR: {tpr_st}, FPR: {fpr_st}, TNR: {tnr_st}, FNR: {fnr_st}")
    print(" ")

  def run(self):
    self.x_train_aug = self.open_csv(self.x_train_aug_file_path)
    self.y_train_aug = list(self.open_csv(self.y_train_aug_file_path)["class"].values)
    self.x_test = self.open_csv(self.x_test_file_path)
    self.y_test = list(self.open_csv(self.y_test_file_path)["class"].values)
    self.SVM()
    self.print_results()
    return self.train_time_svm, self.test_time_svm, self.train_time_pca, self.test_time_pca, self.y_test, self.y_pred, self.accuracy, self.x_train_aug


# parmeter types = pc, kernel, degree, gamma

class RunSVMTrials():
  def __init__(self):
    self.save_folder_path = "/content/drive/MyDrive/asl_project/facial_landmarks_tracking/times/svm/"
    self.trials = 10
    self.parameter_type = "pc"
    self.list_parameters = [4, 8, 12, 16, 20, 24, 28]
    self.kernel = "rbf"
    self.pc = 20
    self.degree = 3
    self.gamma = "auto"
    self.runsvm = RunSVM()
    self.accuracy_df = pd.DataFrame()
    self.train_times_svm_df = pd.DataFrame()
    self.test_times_svm_df = pd.DataFrame()
    self.train_times_pca_df = pd.DataFrame()
    self.test_times_pca_df = pd.DataFrame()
    self.svm_train_save_path = ""
    self.svm_test_save_path = ""
    self.pca_train_save_path = ""
    self.pca_test_save_path = ""
    self.accuracy_save_path = ""

  def run_trials(self):
    for trial in range(self.trials):
      list_accuracy = []
      list_train_times_svm = []
      list_test_times_svm = []
      list_train_times_pca = []
      list_test_times_pca = []
      for parameter in self.list_parameters:
        if self.parameter_type == "pc":
          self.runsvm.kernel = self.kernel
          self.runsvm.n_components = parameter
          self.runsvm.gamma = self.gamma
        if self.parameter_type == "kernel":
          self.runsvm.kernel = parameter
          self.runsvm.n_components = self.pc
          self.runsvm.gamma = self.gamma
        if self.parameter_type == "degree":
          self.runsvm.kernel = "poly"
          self.runsvm.n_components = self.pc
          self.runsvm.degree = parameter
          self.runsvm.gamma = self.gamma
        if self.parameter_type == "gamma":
          self.runsvm.kernel = self.kernel
          self.runsvm.n_components = self.pc
          self.runsvm.degree = self.degree
          self.runsvm.gamma = parameter
        train_time_svm, test_time_svm, train_time_pca, test_time_pca, y_test, y_pred, accuracy, x_train_aug= self.runsvm.run()
        list_accuracy.append(round(accuracy, 3))
        list_train_times_svm.append(round(train_time_svm, 6)*1000)
        list_test_times_svm.append(round(test_time_svm, 6)*1000)
        list_train_times_pca.append(round(train_time_pca, 6)*1000) #in ms
        list_test_times_pca.append(round(test_time_pca, 6)*1000) #in ms
      col_name = "Trial " + str(trial+1)
      self.accuracy_df[col_name] = list_accuracy
      self.train_times_svm_df[col_name] = list_train_times_svm
      self.test_times_svm_df[col_name] = list_test_times_svm
      self.train_times_pca_df[col_name] = list_train_times_pca
      self.test_times_pca_df[col_name] = list_test_times_pca

    avg_train_trial_svm = self.train_times_svm_df.mean(axis=1)
    self.train_times_svm_df["Average"] = round(avg_train_trial_svm, 3) #in ms

    avg_test_trial_svm = self.test_times_svm_df.mean(axis=1)
    self.test_times_svm_df["Average"] = round(avg_test_trial_svm, 3) #in ms

    avg_train_trial_pca = self.train_times_pca_df.mean(axis=1)
    self.train_times_pca_df["Average"] = round(avg_train_trial_pca, 3)  #in ms

    avg_test_trial_pca = self.test_times_pca_df.mean(axis=1)
    self.test_times_pca_df["Average"] = round(avg_test_trial_pca, 3)  #in ms

    self.train_times_svm_df.insert(loc = 0, column = self.parameter_type, value = self.list_parameters)
    self.test_times_svm_df.insert(loc = 0, column = self.parameter_type, value = self.list_parameters)
    self.train_times_pca_df.insert(loc = 0, column = self.parameter_type, value = self.list_parameters)
    self.test_times_pca_df.insert(loc = 0, column = self.parameter_type, value = self.list_parameters)
    self.accuracy_df.insert(loc = 0, column = self.parameter_type, value = self.list_parameters)

  def create_file_paths(self):
    self.list_parameters = [str(parameter) for parameter in self.list_parameters]
    list_parameters_string = "_".join(self.list_parameters)

    if self.parameter_type == "pc":
      svm_train_file_name = "svm_train_times" + "_kernel" + str(self.kernel) + "_degree" + str(self.degree) + "_gamma" + self.gamma + "_pcgrid_" + list_parameters_string + ".csv"
      self.svm_train_save_path = os.path.join(self.save_folder_path, svm_train_file_name)
      svm_test_file_name = "svm_test_times" + "_kernel" + str(self.kernel) + "_degree" + str(self.degree) + "_gamma" + self.gamma + "_pcgrid_" + list_parameters_string + ".csv"
      self.svm_test_save_path = os.path.join(self.save_folder_path, svm_test_file_name)

      pca_train_file_name = "pca_train_times" + "_kernel" + str(self.kernel) + "_degree" + str(self.degree) + "_gamma" + self.gamma + "_pcgrid_" + list_parameters_string + ".csv"
      self.pca_train_save_path = os.path.join(self.save_folder_path, pca_train_file_name)
      pca_test_file_name = "pca_test_times" + "_kernel" + str(self.kernel) + "_degree" + str(self.degree) + "_gamma" + self.gamma + "_pcgrid_" + list_parameters_string + ".csv"
      self.pca_test_save_path = os.path.join(self.save_folder_path, pca_test_file_name)

      accuracy_file_name = "accuracy" + "_kernel" + str(self.kernel) + "_degree" + str(self.degree) + "_gamma" + self.gamma + "_pcgrid_" + list_parameters_string + ".csv"
      self.accuracy_save_path = os.path.join(self.save_folder_path, accuracy_file_name)

    if self.parameter_type == "kernel":
      svm_train_file_name = "svm_train_times" + "_pc" + str(self.pc) + "_degree" + str(self.degree) + "_gamma" + self.gamma + "_kernelgrid_" + list_parameters_string + ".csv"
      self.svm_train_save_path = os.path.join(self.save_folder_path, svm_train_file_name)
      svm_test_file_name = "svm_test_times" + "_pc" + str(self.pc) + "_degree" + str(self.degree) + "_gamma" + self.gamma + "_kernelgrid_" + list_parameters_string + ".csv"
      self.svm_test_save_path = os.path.join(self.save_folder_path, svm_test_file_name)

      pca_train_file_name = "pca_train_times" + "_pc" + str(self.pc) + "_degree" + str(self.degree) + "_gamma" + self.gamma + "_kernelgrid_" + list_parameters_string + ".csv"
      self.pca_train_save_path = os.path.join(self.save_folder_path, pca_train_file_name)
      pca_test_file_name = "pca_test_times" + "_pc" + str(self.pc) + "_degree" + str(self.degree) + "_gamma" + self.gamma + "_kernelgrid_" + list_parameters_string + ".csv"
      self.pca_test_save_path = os.path.join(self.save_folder_path, pca_test_file_name)

      accuracy_file_name = "accuracy" + "_pc" + str(self.pc) + "_degree" + str(self.degree) + "_gamma" + self.gamma + "_kernelgrid_" + list_parameters_string + ".csv"
      self.accuracy_save_path = os.path.join(self.save_folder_path, accuracy_file_name)

    if self.parameter_type == "degree":
      svm_train_file_name = "svm_train_times" + "_pc" + str(self.pc) + "_kernel" + str(self.kernel) + "_gamma" + self.gamma + "_degreegrid_" + list_parameters_string + ".csv"
      self.svm_train_save_path = os.path.join(self.save_folder_path, svm_train_file_name)
      svm_test_file_name = "svm_test_times" + "_pc" + str(self.pc) + "_kernel" + str(self.kernel) + "_gamma" + self.gamma + "_degreegrid_" + list_parameters_string + ".csv"
      self.svm_test_save_path = os.path.join(self.save_folder_path, svm_test_file_name)

      pca_train_file_name = "pca_train_times" + "_pc" + str(self.pc) + "_kernel" + str(self.kernel) + "_gamma" + self.gamma + "_degreegrid_" + list_parameters_string + ".csv"
      self.pca_train_save_path = os.path.join(self.save_folder_path, pca_train_file_name)
      pca_test_file_name = "pca_test_times" + "_pc" + str(self.pc) + "_kernel" + str(self.kernel) + "_gamma" + self.gamma + "_degreegrid_" + list_parameters_string + ".csv"
      self.pca_test_save_path = os.path.join(self.save_folder_path, pca_test_file_name)

      accuracy_file_name = "accuracy" + "_pc" + str(self.pc) + "_kernel" + str(self.kernel) + "_gamma" + self.gamma + "_degreegrid_" + list_parameters_string + ".csv"
      self.accuracy_save_path = os.path.join(self.save_folder_path, accuracy_file_name)

    if self.parameter_type == "gamma":
      svm_train_file_name = "svm_train_times" + "_pc" + str(self.pc) + "_kernel" + str(self.kernel) + "_degree" + str(self.degree) + "_gammagrid_" + list_parameters_string + ".csv"
      self.svm_train_save_path = os.path.join(self.save_folder_path, svm_train_file_name)
      svm_test_file_name = "svm_test_times" + "_pc" + str(self.pc) + "_kernel" + str(self.kernel) + "_degree" + str(self.degree) + "_gammagrid_" + list_parameters_string + ".csv"
      self.svm_test_save_path = os.path.join(self.save_folder_path, svm_test_file_name)

      pca_train_file_name = "pca_train_times" + "_pc" + str(self.pc) + "_kernel" + str(self.kernel) + "_degree" + str(self.degree) + "_gammagrid_" + list_parameters_string + ".csv"
      self.pca_train_save_path = os.path.join(self.save_folder_path, pca_train_file_name)
      pca_test_file_name = "pca_test_times" + "_pc" + str(self.pc) + "_kernel" + str(self.kernel) + "_degree" + str(self.degree) + "_gammagrid_" + list_parameters_string + ".csv"
      self.pca_test_save_path = os.path.join(self.save_folder_path, pca_test_file_name)

      accuracy_file_name = "accuracy" + "_pc" + str(self.pc) + "_kernel" + str(self.kernel) + "_degree" + str(self.degree) + "_gammagrid_" + list_parameters_string + ".csv"
      self.accuracy_save_path = os.path.join(self.save_folder_path, accuracy_file_name)

  def save_results(self):
    self.train_times_svm_df.to_csv(self.svm_train_save_path, index=False)
    self.test_times_svm_df.to_csv(self.svm_test_save_path, index=False)
    self.train_times_pca_df.to_csv(self.pca_train_save_path, index=False)
    self.test_times_pca_df.to_csv(self.pca_test_save_path, index=False)
    self.accuracy_df.to_csv(self.accuracy_save_path, index=False)

  def run(self):
    self.run_trials()
    self.create_file_paths()
    self.save_results()
    return self.svm_train_save_path, self.svm_test_save_path, self.pca_train_save_path, self.pca_test_save_path, self.accuracy_save_path