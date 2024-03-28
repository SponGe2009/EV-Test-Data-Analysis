from PyQt6 import QtGui
from PyQt6.QtWidgets import QLineEdit, QToolTip, QDialog
from PyQt6.QtCore import QPoint, pyqtSignal
from Ui_AddCycleDialog import Ui_Form
from funcs import show_must_read_info


class HoverLineEdit(QLineEdit):
    def __init__(self, tooltip_text="", parent=None):
        super().__init__(parent)
        self.tooltip_text = tooltip_text

    def enterEvent(self, event):
        if self.isEnabled():
            QToolTip.showText(self.mapToGlobal(
                QPoint(20, -20)), self.tooltip_text)
            return super().enterEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        return super().leaveEvent(event)


class AddCycleDialog(QDialog):
    data_transferred = pyqtSignal(list)  # Define the signal at the class level

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Connect the push button's click signal to your custom method
        self.ui.pushButton.clicked.connect(self.transferData)
        self.ui.lineEdit.setValidator(
            QtGui.QDoubleValidator(0, float('inf'), 4))
        self.ui.lineEdit_2.setValidator(
            QtGui.QDoubleValidator(0, float('inf'), 4))

    def transferData(self):
        # This method will be called when the push button is clicked
        if self.ui.lineEdit.text() and self.ui.lineEdit_2.text():
            # Grab data from UI elements
            data = [
                float(self.ui.lineEdit.text()),
                float(self.ui.lineEdit_2.text()),
                self.ui.comboBox.currentText(),
                self.ui.comboBox_2.currentText()
            ]
            if data[1] > data[0]:
                # Emit the signal with collected data
                self.data_transferred.emit(data)
                # Close the dialog
                self.close()
            else:
                show_must_read_info('结束时间必须大于起始时间!')
        else:
            show_must_read_info('请输入起始和结束时间!')
