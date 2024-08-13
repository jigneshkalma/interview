from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np


class Classifier:
    @staticmethod
    def train(x_train2, y_train2):
        model = RandomForestClassifier(n_estimators=100)

        print("Fitting Model..")
        model.fit(x_train2, y_train2)
        print("Model Fitted!")

        return model

    @staticmethod
    def predict(model, x_test2):
        y_predicted = model.predict(x_test2)
        return y_predicted

    @staticmethod
    def test(y_predicted2, y_test2):
        acc = accuracy_score(y_test2, y_predicted2)
        print(f"Accuracy Score of ML model is : {acc}")

    @staticmethod
    def preprocessor():
        mnist = fetch_openml('mnist_784')
        print("Got the data!")
        x, y = mnist['data'], mnist['target']
        x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        arr = np.array(x_test)
        arr1 = arr.reshape(-1, 1)
        print("Starting PCA...")
        pca = PCA(n_components=1)
        x_train = pca.fit_transform(x_train)
        x_test = pca.fit_transform(arr1)
        print("PCA Finished....")
        return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    x_train1, y_train1, x_test1, y_test1 = Classifier.preprocessor()
    model1 = Classifier.train(x_train1, y_train1)
    y_predicted1 = Classifier.predict(model1, x_test1)
    Classifier.test(y_test1, y_predicted1)
