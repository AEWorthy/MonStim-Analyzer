from pathlib import Path
from urllib.parse import unquote, urlsplit


class HelpFileRepository:
    """
    Handles reading help/markdown files for the GUI help system.
    """

    def __init__(self, docs_path: str | Path):
        self.docs_path = Path(docs_path).resolve()

    def _resolve_document_path(self, file: str | Path) -> Path:
        """Return a safe, nested documentation path beneath ``docs_path``."""
        file_path = (self.docs_path / file).resolve()
        try:
            file_path.relative_to(self.docs_path)
        except ValueError as exc:
            raise ValueError(f"Help file is outside the documentation directory: {file}") from exc
        return file_path

    def read_help_file(self, file: str | Path) -> str:
        """Read a bundled Markdown topic from any documentation subdirectory."""
        file_path = self._resolve_document_path(file)
        with open(file_path, encoding="utf-8") as f:
            return f.read()

    def iter_help_files(self):
        """Yield bundled Markdown topics as paths relative to the docs root."""
        yield from (path.relative_to(self.docs_path) for path in self.docs_path.rglob("*.md"))

    def resolve_help_link(self, current_file: str, href: str) -> tuple[str, str] | None:
        """Resolve a relative Markdown link to a bundled help document safely.

        Returns the normalized docs-relative filename and optional anchor. URLs
        outside the docs collection are deliberately left for the caller to
        handle as external links.
        """
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc:
            return None
        try:
            source = self._resolve_document_path(current_file)
        except ValueError:
            return None
        target = (source.parent / unquote(parsed.path)).resolve() if parsed.path else source.resolve()
        try:
            relative_target = target.relative_to(self.docs_path)
        except ValueError:
            return None
        if target.suffix.lower() != ".md" or not target.is_file():
            return None
        return relative_target.as_posix(), unquote(parsed.fragment)
