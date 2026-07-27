import argparse
import logging
import multiprocessing
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler

from PySide6.QtCore import QStandardPaths, QTimer
from PySide6.QtWidgets import QApplication

from monstim_gui.core.splash import SPLASH_INFO

LOG_FILE = "app.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
IS_FROZEN = getattr(sys, "frozen", False)
CONSOLE_DEBUG_MODE = False  # Only relevant if not frozen


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)


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
            filename=log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
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
    logging.getLogger("PySide6").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    get_logger().info(f"Logging to {log_path} (debug={debug})")
    return target_dir


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-dir", metavar="DIR", help="Path to write log files (overrides default)")
    return parser.parse_args()


def exception_hook(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions in Python code."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Call the default excepthook for KeyboardInterrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Log the exception with full details
    logger = get_logger()
    logger.error("=" * 80)
    logger.error("CRITICAL: Uncaught Python exception - Application will terminate")
    logger.error("=" * 80)
    logger.error(f"Exception type: {exc_type.__name__}")
    logger.error(f"Exception value: {exc_value}")
    logger.error(f"Exception module: {exc_type.__module__}")

    # Log the full traceback
    tb_lines = traceback.format_tb(exc_traceback)
    logger.error("Full traceback:")
    for line in tb_lines:
        logger.error(line.rstrip())

    # Also log using exc_info for compatibility
    logger.error("CRITICAL: Uncaught Python exception", exc_info=(exc_type, exc_value, exc_traceback))
    logger.error("=" * 80)

    # Flush log handlers to ensure everything is written
    for handler in logging.getLogger().handlers:
        handler.flush()

    sys.exit(1)


def qt_message_handler(mode, context, message):
    """Handle Qt debug/warning/critical messages."""
    logger = get_logger()
    if mode == 0:  # QtDebugMsg
        logger.debug(f"Qt: {message}")
    elif mode == 1:  # QtWarningMsg
        logger.warning(f"Qt Warning: {message} (file: {context.file}, line: {context.line})")
    elif mode == 2:  # QtCriticalMsg
        logger.error(f"Qt CRITICAL: {message} (file: {context.file}, line: {context.line})")
    elif mode == 3:  # QtFatalMsg
        logger.critical(f"Qt FATAL: {message} (file: {context.file}, line: {context.line})")
        logger.critical("Application will terminate due to Qt fatal error")
    elif mode == 4:  # QtInfoMsg
        logger.info(f"Qt Info: {message}")


def main(is_frozen: bool) -> int:
    try:
        from monstim_gui.core.ui_scaling import setup_dpi_awareness

        setup_dpi_awareness()

        app = QApplication(sys.argv)
        app.setOrganizationName("WorthyLab")
        app.setApplicationName("MonStim Analyzer")
        app.setApplicationVersion(SPLASH_INFO["version"])

        # Install Qt message handler to catch Qt internal errors
        from PySide6.QtCore import qInstallMessageHandler

        qInstallMessageHandler(qt_message_handler)
        get_logger().debug("Qt message handler installed for comprehensive error logging")

        # Reinitialize app_state after QApplication is configured
        from monstim_gui.core.application_state import app_state

        app_state.reinitialize_settings()
        get_logger().info(f"QSettings initialized with org={app.organizationName()}, app={app.applicationName()}")

        if is_frozen:  # Display splash screen if running as a frozen executable.
            from monstim_gui.core.splash import SplashScreen

            splash = SplashScreen()
            splash.show()
            QTimer.singleShot(3000, splash.close)
        gui = MonstimGUI()
        gui.show()
        get_logger().debug("Application launched successfully.")
        return app.exec()

    except Exception as e:
        logger = get_logger()
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

    finally:
        get_logger().info("Application shutting down.")


if __name__ == "__main__":
    args = parse_arguments()
    if IS_FROZEN:
        log_dir = setup_logging(debug=args.debug, log_dir=args.log_dir)
        get_logger().info("Logger initialized. Running via frozen executable.")
    else:
        log_dir = setup_logging(debug=True)
        get_logger().info("Logger initialized. Running via IDE.")
    os.environ["MONSTIM_LOG_DIR"] = log_dir
    sys.excepthook = exception_hook

    # Import the GUI module, matplotlib, and initialize multiprocessing after setting up logging.
    from monstim_gui import MonstimGUI

    multiprocessing.freeze_support()

    get_logger().info("Initialization complete. Starting application.")
    sys.exit(main(IS_FROZEN))
