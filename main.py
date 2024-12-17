import sys
import os
import traceback
import logging
import argparse
import multiprocessing

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Constants
LOG_FILE = 'app.log'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

def setup_logging(debug_mode: bool) -> None:
    log_level = logging.DEBUG if debug_mode else logging.INFO
    if hasattr(sys, '_MEIPASS'):
        logging.basicConfig(filename=os.path.join(sys._MEIPASS, LOG_FILE), level=log_level, format=LOG_FORMAT)
    else:
        logging.basicConfig(filename=LOG_FILE, level=log_level, format=LOG_FORMAT)
    if debug_mode: logging.info("Debug mode enabled")  # noqa: E701
    logging.debug("Logging setup complete")

def setup_matplotlib():
    import matplotlib
    matplotlib.use('QtAgg')  # Ensure compatibility with PyQt6
    logging.getLogger('matplotlib').setLevel(logging.WARNING) # turn off matplotlib debugging messages
    logging.debug("Matplotlib setup complete")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def exception_hook(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

def main(sys_frozen = True) -> int:
    logging.debug("Starting main function")
    try:
        multiprocessing.freeze_support()
        setup_matplotlib()

        logging.debug("Creating QApplication")
        app = QApplication(sys.argv)

        if sys_frozen:
            # Display splash screen
            from monstim_gui.splash import SplashScreen
            splash = SplashScreen()
            splash.show()

            QTimer.singleShot(3000, splash.close)
        
        logging.debug("Creating main window")
        gui = EMGAnalysisGUI()
        logging.debug("Showing main window")
        gui.show()

        logging.debug("Entering main event loop")
        return app.exec()

    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    args = parse_arguments()
    sys_frozen = False
    if getattr(sys, 'frozen', False):
        # If running as a frozen executable, log to file
        sys_frozen = True
        setup_logging(args.debug)
        logging.info("Running as frozen executable")
    else:
        # If running in an IDE, log to console
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
        sys_frozen = False
        logging.info("Running in IDE")

    from monstim_gui import EMGAnalysisGUI
        
    sys.excepthook = exception_hook
    logging.debug("Script running as main")
    sys.exit(main(sys_frozen=sys_frozen))



# To create an executable:
# pyinstaller EMGAnalysis.spec

# To create a debug executable:
# pyinstaller EMGAnalysis.spec --debug