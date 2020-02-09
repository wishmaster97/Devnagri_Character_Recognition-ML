from PyQt5 import QtCore,QtGui,QtWidgets
from Canvas import CanvasWindow
from MainWindowUi import Ui_MainWindow
import sys
import dx
import dxDuplicate

class Driver:
	def __init__(self):


		app = QtWidgets.QApplication(sys.argv)
		MainWindow = QtWidgets.QMainWindow()
		ui = Ui_MainWindow()
		ui.setupUi(MainWindow)
		MainWindow.show()
		
		ui.btnWebCam.clicked.connect(self.webCamRecognition)
		ui.btnCanvas.clicked.connect(self.canvasRecognition)
		ui.btnImage.clicked.connect(self.imageRecognition)

		sys.exit(app.exec_())
		
	def webCamRecognition(self):
		devReg = dx.DevReg();
		devReg.webCam();
		
	def canvasRecognition(self):
		window = CanvasWindow()
		window.setModal(True)
		window.exec_()

	def imageRecognition(self):
		imgReg = dxDuplicate.imgReg();
		imgReg.imageRecog();
		
if __name__ == "__main__":
	Driver()
