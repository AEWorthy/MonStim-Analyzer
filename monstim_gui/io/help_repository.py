import os


class HelpFileRepository:
    """
    Handles reading help/markdown files for the GUI help system.
    """

    def __init__(self, docs_path: str):
        self.docs_path = docs_path

    def read_help_file(self, file: str) -> str:
        file_path = os.path.join(self.docs_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
