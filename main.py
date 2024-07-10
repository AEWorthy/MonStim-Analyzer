import sys
import os
import traceback
import logging
import argparse
from typing import NoReturn
from GUI import EMGAnalysisGUI
from PyQt6.QtWidgets import QApplication
import multiprocessing

# Constants
LOG_FILE = 'app.log'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

def setup_logging(debug_mode: bool) -> None:
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(filename=LOG_FILE, level=log_level, format=LOG_FORMAT)
    if debug_mode:
        print("Debug mode enabled")
    logging.debug("Logging setup complete")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def exception_hook(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

def main() -> int:
    logging.debug("Entering main function")
    try:
        multiprocessing.freeze_support()
        logging.debug("Multiprocessing freeze support called")

        with QApplication(sys.argv) as app:
            gui = EMGAnalysisGUI()
            gui.show()
            logging.debug("GUI shown")
            logging.debug("Entering main event loop")
            return app.exec()
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"An error occurred: {str(e)}")
        return 1

if __name__ == '__main__':
    args = parse_arguments()
    setup_logging(args.debug)
    sys.excepthook = exception_hook
    logging.debug("Script running as main")
    sys.exit(main())

# To create an executable:
# pyinstaller EMGAnalysis.spec