import sys
from qtpy.QtWidgets import QApplication

from .gui import create_automap_widget

def main():
    """Run AutoMap as a standalone application for local development."""
    # Create the application instance
    app = QApplication.instance() or QApplication(sys.argv)

    main_win = create_automap_widget()

    main_win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
