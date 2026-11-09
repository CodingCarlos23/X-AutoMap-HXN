import sys
from qtpy.QtWidgets import QApplication, QMainWindow

from .gui import create_automap_widget

def main():
    """Run AutoMap as a standalone application for local development."""
    app = QApplication.instance() or QApplication(sys.argv)

    widget = create_automap_widget()

    main_win = QMainWindow()
    main_win.setWindowTitle("X-AutoMap")
    main_win.setCentralWidget(widget)
    # Maximize lets the window manager fit the window above the taskbar/panel
    main_win.showMaximized()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
