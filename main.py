import sys
import os
import traceback
import logging
from logging.handlers import RotatingFileHandler
import argparse
import multiprocessing

from PyQt6.QtWidgets import QApplication 
from PyQt6.QtCore import QTimer, QStandardPaths

from monstim_gui.core.splash import SPLASH_INFO

LOG_FILE = 'app.log'
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
IS_FROZEN = getattr(sys, 'frozen', False)
CONSOLE_DEBUG_MODE = False # Only relevant if not frozen


def make_default_log_dir() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if not base:
        base = os.getenv("APPDATA", r"C:\Users\%USERNAME%\AppData\Roaming")
    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def setup_logging(debug: bool, log_dir: str | None = None) -> str:
    target_dir = log_dir or make_default_log_dir()
    if not os.access(target_dir, os.W_OK):
        raise RuntimeError(f"Cannot write to log directory: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    log_path = os.path.join(target_dir, LOG_FILE)

    # Set up the root logger.
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)
    
    # Create a rotating file handler.
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        file_h = RotatingFileHandler(
            filename = log_path, 
            maxBytes=10*1024*1024, 
            backupCount=5,
            encoding='utf-8')
        file_h.setLevel(logging.DEBUG if debug else logging.INFO)
        file_h.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(file_h)
    
    # Create a console handler if in debug mode.
    if debug:
        console_h = logging.StreamHandler()
        if CONSOLE_DEBUG_MODE:
            console_h.setLevel(logging.DEBUG)
        else:
            console_h.setLevel(logging.INFO)
        console_h.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(console_h)

    logging.captureWarnings(True)  # Capture any Python warnings and log them too.
    logging.getLogger("PyQt6").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    root.info(f"Logging to {log_path} (debug={debug})")
    return target_dir

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--log-dir', metavar='DIR', help='Path to write log files (overrides default)')
    return parser.parse_args()

def exception_hook(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.exit(1)

def main(is_frozen : bool) -> int:
    try:
        from monstim_gui.core.ui_scaling import setup_dpi_awareness
        setup_dpi_awareness()
        
        app = QApplication(sys.argv)
        app.setOrganizationName("WorthyLab")
        app.setApplicationName(f"MonStim Analyzer {SPLASH_INFO['version']}")
        app.setApplicationVersion(SPLASH_INFO['version'])

        if is_frozen: # Display splash screen if running as a frozen executable.
            from monstim_gui.core.splash import SplashScreen
            splash = SplashScreen()
            splash.show()
            QTimer.singleShot(3000, splash.close)
        gui = MonstimGUI()
        gui.show()
        logging.info("Application started successfully.")
        return app.exec()

    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        logging.error(traceback.format_exc())
        return 1
    
    finally:
        logging.info("Application shutting down.")

if __name__ == '__main__':
    args = parse_arguments()
    if IS_FROZEN:
        log_dir = setup_logging(debug=args.debug, log_dir=args.log_dir)
        logging.info("Logger initialized. Running via frozen executable.")
    else:
        log_dir = setup_logging(debug=True)
        logging.info("Logger initialized. Running via IDE.")
    os.environ["MONSTIM_LOG_DIR"] = log_dir
    sys.excepthook = exception_hook
    
    # Import the GUI module, matplotlib, and initialize multiprocessing after setting up logging.
    from monstim_gui import MonstimGUI
    multiprocessing.freeze_support()

    logging.info("Initialization complete. Starting application.")
    sys.exit(main(IS_FROZEN))