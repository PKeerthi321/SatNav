import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import DOPGui

app = QApplication(sys.argv)
window = DOPGui()
window.show()
sys.exit(app.exec_())
