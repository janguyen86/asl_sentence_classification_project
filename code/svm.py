from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, List
import time

class RunSVM(object):
  """
  Runs SVM on data.
  """
  def __init__(self):
    self.x_train_aug_file_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\train_all_angle_o9_0.5.csv"
    self.y_train_aug_file_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\train_all_class_o9_0.5.csv"
    self.x_test_file_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\test_angle_o9_0.5.csv"
    self.y_test_file_path = r"C:\Users\nguye\Documents\GitHub\asl_sentence_classification_project\data_csv\test_class_o9_0.5.csv"
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
    """
    Read .csv file as pd.DataFrame
    :param file_path:
    :return: None
    """
    df = pd.read_csv(file_path)
    return df

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
    """
    Calculates the true positive rate, false positive rate, true negative rate, and false negative rate.
    :param class_type: class_type = Class type to calculate metrics for
    :return:
    tpr = True positive rate
    fpr = False positive rate
    tnr = True negative rate
    fnr = False negative rate
    """
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
    """
    Prints the hyperparemeters of the model, the execution times, accuracy and the confusion matrix.
    :return: None
    """
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

  def run(self) -> Tuple[float, float, float, float, List, List, float, List]:
    """
    Runs Random Forest Model and prints out results.
    :return: Model and PCA training and testing times, y_test, y_pred, accuracy, and x_train_aug.
    """
    self.x_train_aug = self.open_csv(self.x_train_aug_file_path)
    self.y_train_aug = list(self.open_csv(self.y_train_aug_file_path)["CLASS"].values)
    self.x_test = self.open_csv(self.x_test_file_path)
    self.y_test = list(self.open_csv(self.y_test_file_path)["CLASS"].values)
    self.SVM()
    self.print_results()
    return self.train_time_svm, self.test_time_svm, self.train_time_pca, self.test_time_pca, self.y_test, self.y_pred, self.accuracy, self.x_train_aug

if __name__=="__main__":
  runsvm = RunSVM()
  train_time_svm, test_time_svm, train_time_pca, test_time_pca, y_test, y_pred, accuracy, x_train_aug = runsvm.run()
