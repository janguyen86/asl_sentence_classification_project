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
    self.n_components = 4
    self.n_estimators = 100
    self.samples_leaf = 4
    self.train_time_pca = 0
    self.test_time_pca = 0
    self.train_time_rf = 0
    self.test_time_rf = 0

  def open_csv(self, file_path: str) -> pd.DataFrame:
    """
    Read .csv file as pd.DataFrame
    :param file_path:
    :return:
    """
    df = pd.read_csv(file_path)
    return df

  def k_fold_validation(self, clf):
    """
    Perform k-fold validation on dataset for k = 5, 7 and 10.
    :param clf:
    :return:
    """
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
    """
    Runs PCA to reduce dimensionality to self.n principal components.
    :param x_train:
    :param x_test:
    :return:
    """
    pca = PCA(n_components=self.n_components, svd_solver = "full")
    self.x_train_aug = pca.fit_transform(self.x_train_aug)
    self.x_test = pca.transform(self.x_test)

  def random_forest_model(self):
    """
    Runs random forest model on inputted data and calculates accuracy and confusion matrix using sklearn algorithms.
    :return: None
    """
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
    self.y_train_aug = list(self.open_csv(self.y_train_aug_file_path)["CLASS"].values)
    self.x_test = self.open_csv(self.x_test_file_path)
    self.y_test = list(self.open_csv(self.y_test_file_path)["CLASS"].values)
    if self.n_components > 0:
      self.run_PCA(self.x_train_aug, self.x_test)
    self.random_forest_model()
    self.print_results()
    return self.train_time_rf, self.test_time_rf, self.train_time_pca, self.test_time_pca, self.y_test, self.y_pred, self.accuracy, self.x_train_aug

if __name__=="__main__":
  runrandomforestmodel = RunRandomForestModel()
  train_time_rf, test_time_rf, train_time_pca, test_time_pca, y_test, y_pred, accuracy, x_train_aug = runrandomforestmodel.run()