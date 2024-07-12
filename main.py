import sys
import os
import traceback
import logging
import argparse
import multiprocessing
import shutil
import atexit
try:
    import pyi_splash # type: ignore
except ModuleNotFoundError:
    pass

from PyQt6.QtWidgets import QApplication

from monstim_gui import EMGAnalysisGUI, SplashScreen




# Constants
LOG_FILE = 'app.log'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

def setup_logging(debug_mode: bool) -> None:
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(filename=LOG_FILE, level=log_level, format=LOG_FORMAT)
    if debug_mode: print("Debug mode enabled")  # noqa: E701
    logging.debug("Logging setup complete")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def exception_hook(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

def cleanup_tempdir():
    if hasattr(sys, '_MEIPASS'):
        temp_dir = sys._MEIPASS
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def main() -> int:
    logging.debug("Starting main function")
    try:
        multiprocessing.freeze_support()

        app = QApplication(sys.argv)

        if 'pyi_splash' in sys.modules:
            pyi_splash.close()  # Close the PyInstaller splash screen
        
        # Display splash screen
        splash = SplashScreen()
        splash.show()

        gui = EMGAnalysisGUI()
        gui.show()

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
    atexit.register(cleanup_tempdir)
    logging.debug("Script running as main")
    sys.exit(main())

# To create an executable:
# pyinstaller EMGAnalysis.spec

# To create a debug executable:
# pyinstaller EMGAnalysis.spec --debug