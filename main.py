import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QApplication,
    QLabel,
    QMessageBox,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QDialog,
)
from torchvision import transforms

from network.AlexNet import AlexNet
from network.ResNet import ResNet
from train_dog_breed import train

BASE_DIR = "models/"

label_dict = dict(
            n02086240='Shih-Tzu',
            n02087394='Rhodesian ridgeback',
            n02088364='Beagle',
            n02089973='English foxhound',
            n02093754='Australian terrier', #
            n02096294='Border terrier',
            n02099601='Golden retriever',
            n02105641='Old English sheepdog', #
            n02111889='Samoyed',
            n02115641='Dingo'
        )
class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setWindowTitle("Dog & Cat network")
        self.setFixedSize(600, 500)
        centralwidget = QWidget(self)
        PredictTab(centralwidget)
        self.setCentralWidget(centralwidget)


class PredictTab(QWidget):
    def __init__(self, parent):
        super(PredictTab, self).__init__(parent)
        self.setFixedSize(600, 500)
        self.imgPath = []
        self.imgIndex = 0
        self.predictions = []
        self.cnn = None
        self.kernel_name = None
        self.n_layers = 1
        self.predict_cat_dog = True
        mainLayout = QVBoxLayout(self)
        self.label_dict = dict(
            n02086240='Shih-Tzu',
            n02087394='Rhodesian ridgeback',
            n02088364='Beagle',
            n02089973='English foxhound',
            n02093754='Australian terrier', #
            n02096294='Border terrier',
            n02099601='Golden retriever',
            n02105641='Old English sheepdog', #
            n02111889='Samoyed',
            n02115641='Dingo'
        )
        self.imgLabel = QLabel()
        self.imgLabel.setStyleSheet(
            "background-color: lightgrey; border: 1px solid gray;"
        )
        self.imgLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.prevButton = QPushButton("<")
        self.prevButton.setMaximumWidth(50)
        self.prevButton.setEnabled(False)
        self.nextButton = QPushButton(">")
        self.nextButton.setMaximumWidth(50)
        self.nextButton.setEnabled(False)
        self.predLabel = QLabel("None")
        self.predLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.predLabel.setFixedWidth(300)
        self.predLabel.setFixedHeight(20)
        hWidget1 = QWidget(self)
        hWidget1.setFixedHeight(20)
        hLayout1 = QHBoxLayout(hWidget1)
        hLayout1.setContentsMargins(0, 0, 0, 0)
        hWidget2 = QWidget(self)
        hWidget2.setFixedHeight(25)
        hLayout2 = QHBoxLayout(hWidget2)
        hLayout2.setContentsMargins(0, 0, 0, 0)
        hWidget3 = QWidget(self)
        hWidget3.setFixedHeight(25)
        hLayout3 = QHBoxLayout(hWidget3)
        hLayout3.setContentsMargins(0, 0, 0, 0)
        hWidget4 = QWidget(self)
        hWidget4.setFixedHeight(25)
        hLayout4 = QHBoxLayout(hWidget4)
        hLayout4.setContentsMargins(0, 0, 0, 0)
        # hWidget.setStyleSheet("border: 1px solid red; padding: 0 0 0 0; margin: 0px;")

        loadButton = QPushButton("Select picture(s)")
        modelButton = QPushButton("Select model (none)")
        predButton = QPushButton("Predict")
        catDogBreedButton = QPushButton("Predict cat or dog")

        loadButton.clicked.connect(self.loadImg)
        self.prevButton.clicked.connect(self.prevImg)
        self.nextButton.clicked.connect(self.nextImg)
        modelButton.clicked.connect(lambda: self.selectedModel(modelButton))
        predButton.clicked.connect(self.predict)
        catDogBreedButton.clicked.connect(lambda: self.selectPredType(catDogBreedButton))

        mainLayout.addWidget(self.imgLabel)
        hLayout1.addWidget(self.prevButton)
        hLayout1.addWidget(self.predLabel)
        hLayout1.addWidget(self.nextButton)
        hLayout2.addWidget(loadButton)
        hLayout2.addWidget(modelButton)
        hLayout3.addWidget(predButton)
        hLayout3.addWidget(catDogBreedButton)
        # hLayout4.addWidget(kernelButton)
        mainLayout.addWidget(hWidget1)
        mainLayout.addWidget(hWidget2)
        mainLayout.addWidget(hWidget3)
        mainLayout.addWidget(hWidget4)

        net = ResNet(num_classes=10, pretrained=True).get_model().cuda()
        train(net, self.label_dict, num_epochs=1)
        self.cnn_dog_breed = net
    def loadImg(self):
        dialog = QFileDialog()
        dialog.setWindowTitle("Select an image")
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setViewMode(QFileDialog.Detail)

        if dialog.exec_():
            self.imgPath = [str(i) for i in dialog.selectedFiles()]
            self.predictions = [None for i in range(len(self.imgPath))]
            self.imgIndex = 0
            print("Selection:")
            for i in self.imgPath:
                print(i)
            self.updateCatDogPixmap(self.imgPath[self.imgIndex])
            self.prevButton.setEnabled(False)
            if len(self.imgPath) > 1:
                self.nextButton.setEnabled(True)
            elif len(self.imgPath) == 1:
                self.nextButton.setEnabled(False)
            self.updateCatDogPixmap(self.imgPath[self.imgIndex])
            # if self.cnn is not None:
            # self.predict()

    def updateCatDogPixmap(self, path, pred=1000):
        self.imgLabel.setPixmap(QtGui.QPixmap(path).scaled(500, 500))
        # self.imgLabel.setScaledContents(True)
        self.predLabel.setText(str(self.predictions[self.imgIndex]))
        if pred < 0.5:
            self.predLabel.setText(
                f"I think it's a Cat! Confidence: {(1 - pred) * 100:.0f}%"
            )
        elif pred > 0.5 and pred != 1000:
            self.predLabel.setText(
                f"I think it's a Dog! Confidence: {pred * 100:.0f}%"
            )
        else:
            self.predLabel.setText("I don't know yet ")

    def predict(self):
        if len(self.imgPath) > 0 and self.cnn is not None:
            try:
                transformer = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
                img = Image.open(self.imgPath[self.imgIndex]).convert('RGB')
                img = transformer(img)
                img = img.unsqueeze(0)
                if self.predict_cat_dog:
                    out = self.cnn(img)
                    probas = F.softmax(out, dim=1).data.cpu().numpy()[0]
                    self.predictions[self.imgIndex] = probas[1]
                    self.updateCatDogPixmap(
                        self.imgPath[self.imgIndex], self.predictions[self.imgIndex]
                    )
                else:
                    out = self.cnn_dog_breed(img.cuda())
                    out_np = out.data.cpu().numpy()
                    best_cluster = np.argmax(out_np, axis=1)
                    idx_label = list(label_dict.keys())
                    pred = label_dict.get(idx_label[best_cluster[0]])
                    proba = F.softmax(out, dim=1).data.cpu().numpy()[0]
                    self.predLabel.setText(
                        f"I think it's a {pred}! Confidence: {proba[best_cluster[0]] * 100:.0f}%"
                    )
            except Exception as e:

                print(e)
                QMessageBox(
                    QMessageBox.Warning,
                    "Error",
                    "Cannot convert image, please select a valid image",
                ).exec_()
        else:
            QMessageBox(
                QMessageBox.Warning,
                "Error",
                "Please select an image and a neural network model before making prediction",
            ).exec_()

    def nextImg(self):
        self.imgIndex += 1
        self.updateCatDogPixmap(self.imgPath[self.imgIndex])
        if self.imgIndex == len(self.imgPath) - 1:
            self.nextButton.setEnabled(False)
        else:
            self.nextButton.setEnabled(True)
        self.prevButton.setEnabled(True)

    def prevImg(self):
        self.imgIndex -= 1
        self.updateCatDogPixmap(self.imgPath[self.imgIndex])
        if self.imgIndex == 0:
            self.prevButton.setEnabled(False)
        else:
            self.prevButton.setEnabled(True)
        self.nextButton.setEnabled(True)

    def selectedModel(self, btn):
        win = ModelWindow()
        if win.exec_():
            name, self.cnn = win.getModel()
            btn.setText(f"Select model ({name})")

    def selectPredType(self, btn):
        if self.predict_cat_dog:
            self.predict_cat_dog = False
            btn.setText("Predict dog breed")
        else:
            self.predict_cat_dog = True
            btn.setText("Predict cat or dog")


class ModelWindow(QDialog):
    def __init__(self):
        super(ModelWindow, self).__init__()
        self.setWindowTitle("Model selection")
        self.model = None
        self.name = None
        mainLayout = QVBoxLayout(self)
        hWidget = QWidget()
        hLayout = QHBoxLayout(hWidget)
        text = QLabel("Select a neural network model: ")
        list = QListWidget()
        self.select = QPushButton("Select")
        self.select.clicked.connect(
            lambda: self.ok_pressed(list.currentItem().text())
        )
        self.delete = QPushButton("Delete")
        self.delete.clicked.connect(lambda: self.delete_pressed(list))
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancel_pressed)

        dir = [
            name
            for name in os.listdir("models")
            if name.endswith(".pt")
        ]

        if len(dir) > 0:
            list.addItems(dir)
        else:
            self.checkCount(list)

        mainLayout.addWidget(text)
        mainLayout.addWidget(list)
        hLayout.addWidget(self.select)
        hLayout.addWidget(self.delete)
        hLayout.addWidget(cancel)
        hLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(hWidget)
        self.setLayout(mainLayout)

    def checkCount(self, list):
        if list.count() == 0:
            list.addItems(["No models found"])
            list.setEnabled(False)
            self.select.setEnabled(False)
            self.delete.setEnabled(False)

    def getModel(self):
        return (self.name, self.model)

    def ok_pressed(self, selected):
        print(selected, "selected")
        try:
            if "alexnet" in selected.lower():
                net = AlexNet(num_classes=2)
                net.load_state_dict(torch.load(BASE_DIR + selected))
            elif "resnet" in selected.lower():
                net = ResNet(num_classes=2, pretrained=False).get_model()
                net.load_state_dict(torch.load(BASE_DIR + selected))
            else:
                net = None
            self.model = net
            self.name = selected
        except Exception as e:
            print("Cannot load model", e)
        self.accept()

    def cancel_pressed(self):
        self.reject()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
