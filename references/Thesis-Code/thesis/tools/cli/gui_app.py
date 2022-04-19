import os
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QImage


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        button = QPushButton('Press me!', self)
        button.clicked.connect(lambda: print('You did it!'))

        self.statusBar().showMessage('Ready')

        exit_act = QAction(QIcon('exit.png'), '&Exit', self)
        exit_act.setShortcut('Ctrl+Q')
        exit_act.setStatusTip('Exit application')
        exit_act.triggered.connect(qApp.quit)

        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(exit_act)

        imp_menu = QMenu('Import', self)
        imp_act = QAction('Import mail', self)
        imp_menu.addAction(imp_act)

        new_act = QAction('New', self)

        file_menu.addAction(new_act)
        file_menu.addMenu(imp_menu)

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exit_act)

        self.resize(800, 600)
        self.center()
        self.show()

    def contextMenuEvent(self, event):
        c_menu = QMenu(self)

        new_act = c_menu.addAction('New')
        open_act = c_menu.addAction('Open')
        quit_act = c_menu.addAction('Quit')
        action = c_menu.exec_(self.mapToGlobal(event.pos()))

        if action == quit_act:
            qApp.quit()

    def center(self):
        geom = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        geom.moveCenter(center_point)
        self.move(geom.topLeft())
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     'Are you sure you want to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # enable high-dpi scaling
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # use high-dpi icons

    w = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()