"""
Pipeline scheduling system for automated execution.

This module provides cron-based scheduling for pipelines.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from src.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from .executor import PipelineExecutor
    from .pipeline import Pipeline

logger = get_logger(__name__)


@dataclass
class ScheduleConfig:
    """Configuration for pipeline schedule."""

    schedule_type: str = "cron"  # 'cron', 'interval', 'once'
    cron_expression: str | None = None  # e.g., "0 0 * * *" for daily at midnight
    interval_seconds: int | None = None  # For interval-based scheduling
    start_time: datetime | None = None  # For one-time execution
    enabled: bool = True
    max_retries: int = 0
    retry_delay_seconds: int = 60


@dataclass
class ScheduledPipeline:
    """A pipeline with associated schedule."""

    pipeline_id: str
    pipeline: Pipeline
    schedule: ScheduleConfig
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0
    failure_count: int = 0
    is_running: bool = False


class PipelineScheduler:
    """
    Schedule and manage automated pipeline execution.

    Supports cron-based scheduling, interval-based scheduling,
    and one-time execution.
    """

    def __init__(self, executor: PipelineExecutor) -> None:
        """
        Initialize pipeline scheduler.

        Args:
            executor: PipelineExecutor instance for running pipelines
        """
        self.executor = executor
        self.scheduled_pipelines: dict[str, ScheduledPipeline] = {}
        self.logger = get_logger(__name__)
        self._running = False
        self._scheduler_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def schedule_pipeline(
        self,
        pipeline: Pipeline,
        schedule: ScheduleConfig,
    ) -> None:
        """
        Schedule a pipeline for automated execution.

        Args:
            pipeline: Pipeline to schedule
            schedule: Schedule configuration
        """
        with self._lock:
            scheduled = ScheduledPipeline(
                pipeline_id=pipeline.pipeline_id,
                pipeline=pipeline,
                schedule=schedule,
                next_run=self._calculate_next_run(schedule),
            )

            self.scheduled_pipelines[pipeline.pipeline_id] = scheduled
            self.logger.info(
                f"Scheduled pipeline {pipeline.name} (next run: {scheduled.next_run})"
            )

    def unschedule_pipeline(self, pipeline_id: str) -> bool:
        """
        Remove a pipeline from the schedule.

        Args:
            pipeline_id: ID of pipeline to unschedule

        Returns:
            True if pipeline was unscheduled, False if not found
        """
        with self._lock:
            if pipeline_id in self.scheduled_pipelines:
                del self.scheduled_pipelines[pipeline_id]
                self.logger.info(f"Unscheduled pipeline: {pipeline_id}")
                return True

            self.logger.warning(f"Pipeline {pipeline_id} not found in schedule")
            return False

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            self.logger.warning("Scheduler is already running")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        self.logger.info("Pipeline scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            self.logger.warning("Scheduler is not running")
            return

        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        self.logger.info("Pipeline scheduler stopped")

    def _scheduler_loop(self) -> None:
        """Main scheduler loop that checks and executes pipelines."""
        while self._running:
            try:
                current_time = datetime.now()

                with self._lock:
                    for scheduled in list(self.scheduled_pipelines.values()):
                        if not scheduled.schedule.enabled:
                            continue

                        if scheduled.is_running:
                            continue

                        if scheduled.next_run and current_time >= scheduled.next_run:
                            # Time to run the pipeline
                            self._execute_scheduled_pipeline(scheduled)

                # Sleep for a short time before next check
                time.sleep(10)

            except Exception as e:
                self.logger.exception(f"Error in scheduler loop: {e}")
                time.sleep(60)  # Wait longer on errors

    def _execute_scheduled_pipeline(self, scheduled: ScheduledPipeline) -> None:
        """
        Execute a scheduled pipeline.

        Args:
            scheduled: The scheduled pipeline to execute
        """
        scheduled.is_running = True
        scheduled.last_run = datetime.now()

        def on_complete() -> None:
            """Callback when pipeline completes."""
            with self._lock:
                scheduled.is_running = False
                scheduled.run_count += 1

                # Check execution result
                result = self.executor.get_result(scheduled.pipeline_id)
                if result and result.error:
                    scheduled.failure_count += 1
                    self.logger.error(
                        f"Scheduled pipeline {scheduled.pipeline.name} failed: {result.error}"
                    )
                else:
                    self.logger.info(
                        f"Scheduled pipeline {scheduled.pipeline.name} completed successfully"
                    )

                # Calculate next run time
                scheduled.next_run = self._calculate_next_run(scheduled.schedule)
                self.logger.info(
                    f"Next run for {scheduled.pipeline.name}: {scheduled.next_run}"
                )

        # Execute pipeline asynchronously
        try:
            self.logger.info(f"Executing scheduled pipeline: {scheduled.pipeline.name}")
            self.executor.execute(scheduled.pipeline, async_mode=True)

            # Start a thread to monitor completion
            def monitor() -> None:
                while self.executor.is_running(scheduled.pipeline_id):
                    time.sleep(1)
                on_complete()

            threading.Thread(target=monitor, daemon=True).start()

        except Exception as e:
            self.logger.exception(f"Failed to execute scheduled pipeline: {e}")
            scheduled.is_running = False
            scheduled.failure_count += 1
            scheduled.next_run = self._calculate_next_run(scheduled.schedule)

    def _calculate_next_run(self, schedule: ScheduleConfig) -> datetime | None:
        """
        Calculate the next run time based on schedule configuration.

        Args:
            schedule: Schedule configuration

        Returns:
            Next run time, or None if no future runs
        """
        if schedule.schedule_type == "once":
            return schedule.start_time

        if schedule.schedule_type == "interval" and schedule.interval_seconds:
            return datetime.now().timestamp() + schedule.interval_seconds

        if schedule.schedule_type == "cron" and schedule.cron_expression:
            # Simple cron parsing (basic implementation)
            # For production, use a library like croniter
            return self._parse_cron_next_run(schedule.cron_expression)

        return None

    def _parse_cron_next_run(self, cron_expression: str) -> datetime:
        """
        Parse cron expression and calculate next run time.

        This is a simplified implementation. For production use,
        consider using a library like croniter.

        Args:
            cron_expression: Cron expression (e.g., "0 0 * * *")

        Returns:
            Next run datetime
        """
        # Simplified: just parse basic interval patterns
        # Format: minute hour day month weekday
        parts = cron_expression.split()

        if len(parts) != 5:  # noqa: PLR2004
            self.logger.warning(f"Invalid cron expression: {cron_expression}")
            # Default to daily at midnight
            from datetime import timedelta

            now = datetime.now()
            next_run = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run

        # Try to parse using croniter if available
        try:
            from croniter import croniter

            cron = croniter(cron_expression, datetime.now())
            return cron.get_next(datetime)
        except ImportError:
            self.logger.warning(
                "croniter not installed, using simplified cron parsing"
            )
            # Fallback to simple daily schedule
            from datetime import timedelta

            now = datetime.now()
            next_run = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run

    def get_schedule(self, pipeline_id: str) -> ScheduledPipeline | None:
        """
        Get schedule information for a pipeline.

        Args:
            pipeline_id: ID of the pipeline

        Returns:
            ScheduledPipeline if scheduled, None otherwise
        """
        return self.scheduled_pipelines.get(pipeline_id)

    def list_scheduled_pipelines(self) -> list[ScheduledPipeline]:
        """
        Get list of all scheduled pipelines.

        Returns:
            List of ScheduledPipeline objects
        """
        return list(self.scheduled_pipelines.values())

    def is_running(self) -> bool:
        """
        Check if scheduler is running.

        Returns:
            True if scheduler is running
        """
        return self._running
