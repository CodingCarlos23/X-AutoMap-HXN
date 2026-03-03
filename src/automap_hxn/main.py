import sys
from PyQt5.QtWidgets import QApplication

from app_state import AppState
from gui import MainWindow

if __name__ == '__main__':
    # Create the application instance
    app = QApplication(sys.argv)

    # Create the application state object
    app_state = AppState()

    # Create the main window, passing the state to it
    main_win = MainWindow(app_state)

    # Show the window and start the event loop
    main_win.show()
    sys.exit(app.exec_())
