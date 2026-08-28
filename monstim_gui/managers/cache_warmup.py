"""Cancelable GUI coordination for domain cache preparation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Event

from PySide6.QtCore import QElapsedTimer, QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import QProgressDialog

from monstim_gui.core.application_state import app_state

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WarmTask:
    target: object
    products: frozenset[str]
    methods: tuple[str, ...]
    detail: str
    jobs: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = ()


class CacheWarmUpWorker(QThread):
    progress = Signal(int, int, str)
    failed = Signal(str)

    def __init__(self, tasks: tuple[WarmTask, ...], parent=None):
        super().__init__(parent)
        self.tasks = tasks
        self._cancel = Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        total_jobs = sum(len(task.jobs) if task.jobs else 1 for task in self.tasks)
        total = max(1, total_jobs * 1000)
        job_index = 0
        try:
            for task in self.tasks:
                if self._cancel.is_set():
                    break
                jobs = task.jobs or ((task.products, task.methods),)
                for products, methods in jobs:
                    if self._cancel.is_set():
                        break
                    self.progress.emit(job_index * 1000, total, task.detail)

                    def report(done, count, detail, offset=job_index):
                        fraction = (done / count) if count else 0.0
                        self.progress.emit(offset * 1000 + int(1000 * fraction), total, detail)

                    task.target.prepare_cache(
                        products,
                        methods,
                        report,
                        self._cancel.is_set,
                    )
                    if not self._cancel.is_set():
                        job_index += 1
                        self.progress.emit(job_index * 1000, total, task.detail)
        except Exception as exc:
            logger.exception("Plot cache warm-up failed")
            self.failed.emit(str(exc))


class CacheWarmUpCoordinator(QObject):
    """Merge enabled policies into a prioritized, deduplicated work queue."""

    def __init__(self, gui):
        super().__init__(gui)
        self.gui = gui
        self.worker: CacheWarmUpWorker | None = None
        self.dialog: QProgressDialog | None = None
        self._scheduled = False
        self._elapsed = QElapsedTimer()
        self._last_progress = (0, 1, "Starting...")
        self._dialog_timer = QTimer(self)
        self._dialog_timer.setInterval(1000)
        self._dialog_timer.timeout.connect(self._refresh_dialog)

    def build_tasks(self) -> tuple[WarmTask, ...]:
        experiment = self.gui.current_experiment
        current_dataset = self.gui.current_dataset
        current_session = self.gui.current_session
        policy = app_state.get_load_policy()
        session_work: dict[object, dict[str | None, set[str]]] = {}

        def add_session(session, level_policy) -> None:
            if not (level_policy.filtered_signals or level_policy.methods or level_policy.aggregates or level_policy.prepare_mmax):
                return
            jobs = session_work.setdefault(session, {})
            if level_policy.filtered_signals:
                jobs.setdefault(None, set()).add("filtered_signals")
            for method in level_policy.methods:
                jobs.setdefault(method, set()).add("window_results")
            aggregate_methods = level_policy.methods or (getattr(session, "default_method", "rms"),)
            if level_policy.aggregates:
                for method in aggregate_methods:
                    jobs.setdefault(method, set()).add("amplitudes")
            if level_policy.prepare_mmax:
                for method in aggregate_methods:
                    jobs.setdefault(method, set()).add("mmax")

        if policy.session.enabled and current_session is not None:
            add_session(current_session, policy.session)
        if policy.dataset.enabled and current_dataset is not None:
            for session in current_dataset.sessions:
                add_session(session, policy.dataset)
        if policy.experiment.enabled and experiment is not None:
            for dataset in experiment.datasets:
                for session in dataset.sessions:
                    add_session(session, policy.experiment)

        ordered_sessions = []
        if current_session in session_work:
            ordered_sessions.append(current_session)
        if current_dataset is not None:
            ordered_sessions.extend(session for session in current_dataset.sessions if session in session_work and session not in ordered_sessions)
        if experiment is not None:
            ordered_sessions.extend(
                session
                for dataset in experiment.datasets
                for session in dataset.sessions
                if session in session_work and session not in ordered_sessions
            )
        tasks = []
        for session in ordered_sessions:
            work = session_work[session]
            filtered = work.get(None, set())
            method_jobs = []
            for index, method in enumerate(sorted(key for key in work if key is not None)):
                products = set(work[method])
                if index == 0:
                    products.update(filtered)
                method_jobs.append((frozenset(products), (method,)))
            if not method_jobs and filtered:
                method_jobs.append((frozenset(filtered), ()))
            products = frozenset(product for job_products, _methods in method_jobs for product in job_products)
            methods = tuple(method for _products, job_methods in method_jobs for method in job_methods)
            tasks.append(WarmTask(session, products, methods, f"Session {session.id}", tuple(method_jobs)))

        dataset_work: dict[object, set[str]] = {}
        if policy.dataset.enabled and policy.dataset.aggregates and current_dataset is not None:
            dataset_work.setdefault(current_dataset, set()).update(policy.dataset.methods)
        if policy.experiment.enabled and policy.experiment.aggregates and experiment is not None:
            for dataset in experiment.datasets:
                dataset_work.setdefault(dataset, set()).update(policy.experiment.methods)
        ordered_datasets = []
        if current_dataset in dataset_work:
            ordered_datasets.append(current_dataset)
        if experiment is not None:
            ordered_datasets.extend(dataset for dataset in experiment.datasets if dataset in dataset_work and dataset not in ordered_datasets)
        for dataset in ordered_datasets:
            tasks.append(
                WarmTask(
                    dataset,
                    frozenset({"dataset_aggregates"}),
                    tuple(sorted(dataset_work[dataset])),
                    f"Dataset {dataset.id} aggregates",
                )
            )
        if policy.experiment.enabled and policy.experiment.aggregates and experiment is not None:
            tasks.append(
                WarmTask(
                    experiment,
                    frozenset({"experiment_aggregates"}),
                    tuple(policy.experiment.methods),
                    f"Experiment {experiment.id} aggregates",
                )
            )
        return tuple(tasks)

    def request(self) -> None:
        """Coalesce selection events and start only after the window is visible."""
        if self._scheduled:
            return
        self._scheduled = True
        QTimer.singleShot(0, self._start_scheduled)

    def _start_scheduled(self) -> None:
        if not self.gui.isVisible():
            self._scheduled = False
            return
        self._scheduled = False
        self.cancel_and_wait()
        tasks = self.build_tasks()
        if not tasks:
            return
        worker = CacheWarmUpWorker(tasks, self)
        self.worker = worker
        self._elapsed.start()
        worker.progress.connect(self._on_progress)
        worker.failed.connect(lambda message: logger.warning("Cache warm-up stopped: %s", message))
        worker.finished.connect(self._on_finished)
        worker.start()
        QTimer.singleShot(500, lambda expected=worker: self._show_dialog(expected))

    def _show_dialog(self, expected: CacheWarmUpWorker) -> None:
        if self.worker is not expected or not expected.isRunning():
            return
        dialog = QProgressDialog("Preparing plot cache...", "Cancel", 0, len(expected.tasks), self.gui)
        dialog.setWindowTitle("Plot Cache Warm-Up")
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.canceled.connect(expected.request_cancel)
        self.dialog = dialog
        dialog.show()
        self._dialog_timer.start()

    def _on_progress(self, completed: int, total: int, detail: str) -> None:
        self._last_progress = (completed, total, detail)
        self._refresh_dialog()

    def _refresh_dialog(self) -> None:
        if self.dialog is not None:
            completed, total, detail = self._last_progress
            self.dialog.setMaximum(max(1, total))
            self.dialog.setValue(completed)
            elapsed_seconds = self._elapsed.elapsed() // 1000
            self.dialog.setLabelText(f"Preparing plot cache...\n{detail}\nElapsed: {elapsed_seconds // 60}:{elapsed_seconds % 60:02d}")

    def _on_finished(self) -> None:
        self._dialog_timer.stop()
        if self.dialog is not None:
            self.dialog.close()
            self.dialog.deleteLater()
            self.dialog = None
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()

    def cancel_and_wait(self) -> None:
        worker = self.worker
        if worker is None:
            return
        worker.request_cancel()
        worker.wait()
        if self.worker is worker:
            self._on_finished()
