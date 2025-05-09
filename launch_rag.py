import sys, os
from PyQt6.QtWidgets import QApplication
from core.rag_core import RAGApp
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RAGApp()
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())
