# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Benchmarker utility for profiling operation timing in AIProgram.

This module provides a framework for benchmarking ML model implementations by
measuring the execution time of each operation using the Profiler API from
standalone_swift and coreai-runtime.

Key components:
- Benchmarker: Main class for collecting and reporting operation timing
- BenchmarkResult: Data class containing timing information for operations
- CoreAIBenchmarker: Benchmarker implementation using Core AI Runtime
"""

import asyncio
import json
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TextIO

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir
import numpy as np
from coreai._compiler.ir import Operation, WalkResult
from coreai.authoring import AIProgram
from coreai.runtime import (
    AIModel,
    LogEvent,
    NDArray,
    Profiler,
    SpecializationOptions,
)
from typing_extensions import Self

from .annotations import _Annotation, _AnnotationCallback, _AnnotationLine
from .debug_info import (
    DebugInfoRecord,
    _build_compile_id_to_coreai_map,
    _build_delegate_op_to_source_map,
    get_operation_id,
    parse_debug_infos,
)
from .source_annotator import _annotate_operations
from .table_writer import (
    _Column,
    _Row,
    _TableSpec,
    _TreeNode,
    _write_table,
    _write_tree,
)
from .utils import LocationInfo, get_operation_locations, split_module_frame

logger = logging.getLogger(__name__)


def _get_default_excluded_operations() -> tuple[str, ...]:
    """
    Get default tuple of operation names to exclude from timing measurements.

    Returns:
        Tuple of operation names that should not be timed by default

    """
    return (
        "coreai.graph",
        "coreai.constant",
    )


class _LogEventPhase(Enum):
    """Phase of a profiling event."""

    LOAD = "load"
    """Model loading phase."""

    COMPILE = "compile"
    """Compilation phase."""

    INFERENCE = "inference"
    """Inference/execution phase."""

    UNKNOWN = "unknown"
    """Unknown or unrecognized phase."""

    @classmethod
    def _missing_(cls, value: object) -> "_LogEventPhase":
        """Return UNKNOWN for unrecognized phase values."""
        return cls.UNKNOWN


class _BenchmarkerState(Enum):
    """State of the benchmarker."""

    LOADING = "loading"
    """Loading and initializing model."""

    RUNNING = "running"
    """Actively running benchmark iterations."""

    COMPLETED = "completed"
    """Benchmark completed."""


@dataclass(frozen=True)
class Statistics:
    """Statistical summary of measurements."""

    minimum: float
    """Minimum value."""

    maximum: float
    """Maximum value."""

    average: float
    """Average (mean) value."""

    std_dev: float
    """Standard deviation."""

    median: float
    """Median value."""

    @staticmethod
    def from_values(values: list[float]) -> "Statistics | None":
        """
        Create Statistics from a list of values.

        Args:
            values: List of numeric values

        Returns:
            Statistics object or None if values is empty

        """
        if len(values) == 0:
            return None

        minimum = float(np.min(values))
        maximum = float(np.max(values))
        average = float(np.mean(values))
        std_dev = float(np.std(values))
        median = float(np.median(values))

        return Statistics(
            minimum=minimum,
            maximum=maximum,
            average=average,
            std_dev=std_dev,
            median=median,
        )


@dataclass(frozen=True)
class Measurement:
    """Measurement containing statistics and raw samples."""

    statistics: Statistics | None
    """Statistical summary of the samples."""

    samples: list[float]
    """Raw sample values."""

    @staticmethod
    def from_samples(samples: list[float]) -> "Measurement":
        """
        Create Measurement from a list of samples.

        Args:
            samples: List of sample values

        Returns:
            Measurement object with computed statistics

        """
        return Measurement(
            statistics=Statistics.from_values(values=samples),
            samples=samples,
        )

    @property
    def sort_key(self) -> tuple[bool, float | None]:
        """
        Get sort key for this measurement.

        Returns:
            Tuple of (has_statistics, median_value) for sorting

        """
        return (
            self.statistics is not None,
            self.statistics.median if self.statistics is not None else None,
        )


@dataclass(frozen=True)
class _TimingAnnotation:
    """
    A dispatch's timing as an :class:`_Annotation`.

    Carries the numbers rather than a formatted string, so a consumer that renders
    its own output gets them from :meth:`data` instead of parsing text back.
    """

    op_ids: tuple[int, ...]
    """Core AI operation IDs the dispatch covered."""

    op_names: tuple[str, ...]
    """Names of those operations."""

    median_ms: float
    """Median duration of the dispatch, in milliseconds."""

    average_ms: float
    """Mean duration of the dispatch, in milliseconds."""

    samples: int
    """How many measurements the statistics are over."""

    def lines(self: Self) -> "tuple[_AnnotationLine, ...]":
        """
        Return this timing as a single line.

        Returns:
            One :class:`_AnnotationLine` naming the operations and their timing.

        """
        fused = "+".join(self.op_names) or "unknown"
        return (
            _AnnotationLine(
                f"{fused}: {self.average_ms:.3f}ms (med: {self.median_ms:.3f}ms)",
                style="",
            ),
        )

    def data(self: Self) -> dict[str, Any]:
        """
        Return the timing as plain values.

        Returns:
            The op ids and names the duration covers, and the duration itself. The
            ids matter as much as the numbers: the duration belongs to the whole
            dispatch, so a consumer must not add it up per operation.

        """
        return {
            "op_ids": list(self.op_ids),
            "op_names": list(self.op_names),
            "median_ms": self.median_ms,
            "average_ms": self.average_ms,
            "samples": self.samples,
        }


@dataclass
class OperationTiming:
    """
    Timing information for a fused group of operations.

    The runtime profiler measures a compile identifier -- a possibly-fused group of
    Core AI ops -- as a whole. The recorded ``measurement`` is the timing of that
    whole group, and ``op_ids`` lists every Core AI op it contains.
    """

    op_ids: list[int]
    """Core AI operation IDs that make up this fused group."""

    operations: list[Operation]
    """The Core AI operations that make up this fused group.

    A view into the module the ``AIProgram`` owns, so it is valid only while that
    program is alive; extract what is needed rather than stashing the result."""

    measurement: Measurement
    """Measurement containing statistics and timing samples in milliseconds."""

    @property
    def op_id(self) -> int:
        """Representative (first) operation ID of the group."""
        return self.op_ids[0]

    @property
    def op_names(self) -> list[str]:
        """Names of the Core AI operations in this group."""
        return [op.name for op in self.operations]

    def to_row(self: Self) -> _Row:
        """
        Return this dispatch's cells for the summary table.

        Every Core AI op id it covers is listed, not a representative: those
        collide -- four rows read "121" on one model -- leaving rows that measured
        different work indistinguishable. The ids and names fold within their
        columns, so a wide fused group stays fully described.

        Returns:
            The :class:`_Row` describing this dispatch.

        """
        statistics = self.measurement.statistics
        numbers = (
            (
                f"{statistics.median:.6f}",
                f"{statistics.average:.6f}",
                f"{statistics.minimum:.6f}",
                f"{statistics.maximum:.6f}",
                f"{statistics.std_dev:.6f}",
            )
            if statistics is not None
            else ("N/A",) * 5
        )
        return _Row(
            cells=(
                ", ".join(str(op_id) for op_id in self.op_ids),
                "\n".join(self.op_names) or "unknown",
                *numbers,
            ),
        )

    @property
    def tree_label(self: Self) -> str:
        """One line naming this dispatch's operations and its timing."""
        fused = "+".join(self.op_names) or "unknown"
        statistics = self.measurement.statistics
        if statistics is None:
            return f"{fused}: no timing data"
        return (
            f"{fused}: med {statistics.median:.6f}ms, avg {statistics.average:.6f}ms, "
            f"min {statistics.minimum:.6f}ms, max {statistics.maximum:.6f}ms"
        )


@dataclass
class ModuleTiming:
    """
    Timing information for a module and its operations.

    Represents a hierarchical grouping of operations by their module path.
    """

    name: str
    """Module name as the stack trace gives it, instance number included:
    ``Linear$3``. Also the key :meth:`BenchmarkResult.get_module_timings` files this
    module under, so it identifies the instance rather than the type."""

    operation_timings: list[OperationTiming]
    """Operation timings (one per fused group) belonging to this module."""

    children: list["ModuleTiming"]
    """Child modules."""

    @property
    def type_name(self: Self) -> str:
        """
        The module's type, without the instance number.

        Returns:
            ``"Linear"`` for ``Linear$3``, and the whole name when it carries no
            instance number.

        """
        return split_module_frame(self.name)[0]

    @property
    def instance(self: Self) -> int | None:
        """
        Which instance of :attr:`type_name` this module is.

        Returns:
            ``3`` for ``Linear$3``. None when the name carries no instance number,
            as ``<unknown>`` does.

        """
        return split_module_frame(self.name)[1]

    # A module deliberately reports no total. Fusion crosses module boundaries --
    # 86% of dispatches in a 3-layer MLP and 93% in a transformer block cover more
    # than one module -- and a dispatch is attributed whole to the module of its
    # first member, so a total charges one module for a sibling's work and leaves
    # the sibling reading 0.000ms. A LayerNorm fused into its neighbour is not free,
    # and saying so was worse than saying nothing.
    #
    # :meth:`get_all_operations` still exposes the dispatches, so a caller that
    # knows a group spans modules can aggregate on its own terms.

    def get_all_operations(self: Self) -> list[OperationTiming]:
        """
        Get all operation timings in this module and its children recursively.

        Returns:
            List of all OperationTiming objects

        """
        all_ops = list(self.operation_timings)
        for child in self.children:
            all_ops.extend(child.get_all_operations())
        return all_ops

    def get_all_modules(self: Self) -> list["ModuleTiming"]:
        """
        Get a flattened list of this module and all its children recursively.

        Returns:
            List of all ModuleTiming objects (this module and all descendants)

        """
        all_modules: list[ModuleTiming] = [self]
        for child in self.children:
            all_modules.extend(child.get_all_modules())
        return all_modules

    def get_operations_at_location(
        self: Self,
        location: LocationInfo,
    ) -> list[OperationTiming]:
        """
        Find operation timings at a specific source location.

        Searches this module and all children recursively for timing groups any of
        whose operations have the given file/line/col location.

        Args:
            location: LocationInfo to search for

        Returns:
            List of OperationTiming objects matching the location

        """
        matching_ops = []

        # Search all operation timings in this module and children
        for timing in self.get_all_operations():
            matched = False
            for operation in timing.operations:
                for op_loc in get_operation_locations(operation):
                    if (
                        op_loc.filename == location.filename
                        and op_loc.line == location.line
                        and op_loc.col == location.col
                    ):
                        matching_ops.append(timing)
                        matched = True
                        break
                if matched:
                    break  # Don't add the same timing twice

        return matching_ops

    def _annotation_callback(self: Self) -> _AnnotationCallback:
        """
        Build the callback describing each operation's timing.

        A group's timing is reported once, on its representative operation. Other
        members return ``None`` and are skipped: the runtime measured the group as a
        whole, so there is no per-member figure to give, and annotating each one
        repeated a single measurement as many times as the group had members --
        several identical comments stacked above one source line.

        Returns:
            A callback mapping an operation to its annotation, or ``None`` for an
            operation this module did not time, or that a group already covers.

        """
        timing_by_representative: dict[int, OperationTiming] = {
            timing.op_id: timing for timing in self.get_all_operations()
        }

        def annotate(operation: Operation) -> _Annotation | None:
            operation_id = get_operation_id(operation)
            if operation_id is None:
                return None

            # Only the representative carries the group's timing; other members of
            # the same group are covered by it and skipped.
            timing = timing_by_representative.get(operation_id)
            if timing is None:
                return None

            statistics = timing.measurement.statistics
            if statistics is None:
                return None

            return _TimingAnnotation(
                op_ids=tuple(timing.op_ids),
                op_names=tuple(timing.op_names),
                median_ms=statistics.median,
                average_ms=statistics.average,
                samples=len(timing.measurement.samples),
            )

        return annotate

    def annotate_dominant_source(
        self: Self,
        output: TextIO,
        exclude: Callable[[LocationInfo], bool] | None = None,
    ) -> None:
        """
        Find the dominant source file and annotate it with timing information.

        Uses operations from this module and its children, and renders through the
        shared source annotator, so styling follows the destination stream -- colour
        for a terminal, plain text for a file -- and each comment is indented to
        match the line it describes.

        Args:
            output: Text stream to write annotated source to (file or stdout)
            exclude: Optional callable to filter out locations. If None, uses default
                    which excludes torch files, exported_program.py, and "-"

        """
        operations = [
            operation
            for timing in self.get_all_operations()
            for operation in timing.operations
        ]
        _annotate_operations(
            operations,
            self._annotation_callback(),
            output,
            exclude=exclude,
        )

    def to_tree_node(self: Self, show_operations: bool = False) -> _TreeNode:
        """
        Build this module's node, with its children beneath it.

        Args:
            show_operations: Whether to list each dispatch under its module
                (default: False)

        Returns:
            The :class:`_TreeNode` for this module.

        """
        # A count, not a duration: see the note above :meth:`get_all_operations`.
        node = _TreeNode(
            label=(
                f"{self.name} ({len(self.operation_timings)} "
                f"dispatch{'es' if len(self.operation_timings) != 1 else ''})"
                if self.operation_timings
                else self.name
            )
        )

        if show_operations:
            for timing in self.operation_timings:
                node.add(timing.tree_label)

        for child in self.children:
            node.add(child.to_tree_node(show_operations=show_operations))

        return node

    def write_to(
        self: Self,
        output: TextIO,
        show_operations: bool = False,
        *,
        width: int | None = None,
    ) -> None:
        """
        Write this module and its children to output as a tree.

        Rendered through the shared tree writer, so a long fused name wraps within
        the width its depth leaves rather than being truncated.

        Args:
            output: Text stream to write to
            show_operations: Whether to list each dispatch (default: False)
            width: Console width to render at. Defaults to
                :data:`~coreai_torch.debugging.table_writer._DEFAULT_WIDTH`.

        """
        _write_tree(
            self.to_tree_node(show_operations=show_operations), output, width=width
        )


@dataclass
class BenchmarkResult:
    """
    Result of benchmarking an AIProgram.

    Contains timing information for each operation, organized by operation ID.
    """

    operation_timings: list[OperationTiming]
    """
    List of operation timings, one per profiled fused group of operations.
    """

    operations_by_id: dict[int, Operation] = field(default_factory=dict)
    """
    Every Core AI operation in the benchmarked function, by id -- including those no
    dispatch reported.

    The denominator for coverage, and which operations count is the caller's
    judgement: constants and layout operations such as ``pad`` or ``broadcast_to``
    fold into a consumer's addressing and never become GPU work, so counting them
    understates coverage badly. An operation absent from every
    :attr:`OperationTiming.op_ids` was not timed, and its location says where in the
    source it came from.
    """

    # No per-operation lookup is offered. A dispatch's duration belongs to every
    # operation in it, so returning that duration for one op id -- which
    # `get_average_timing` and `get_measurement` used to do -- reads as the cost of
    # that operation, and summing over ops then multiplies the real cost by each
    # group's width. A caller who wants the dispatch an operation landed in can find
    # it, and sees `op_ids` when they do:
    #
    #     op_id = get_operation_id(operation)
    #     timing = next(
    #         (t for t in result.operation_timings if op_id in t.op_ids), None
    #     )

    def get_operation_summary(self: Self) -> list[OperationTiming]:
        """
        Get operation timings sorted by median duration.

        Returns:
            List of OperationTiming sorted by median duration (descending)

        """
        summary = list(self.operation_timings)

        # Sort by median duration (descending)
        summary.sort(key=lambda timing: timing.measurement.sort_key, reverse=True)
        return summary

    def get_module_timings(self) -> dict[str, ModuleTiming]:
        """
        Group dispatches by module, from their operations' stack traces.

        Each stack-trace frame becomes a nested module. A dispatch is filed against
        the deepest module containing *every* operation it covers, so a dispatch
        listed under a module is wholly inside it. Filing it against its first
        member's module instead charged one module for a sibling's work -- and fusion
        crosses module boundaries in most dispatches, so that was the common case,
        not the exception.

        Modules appear even when no dispatch is filed against them, which is the
        honest reading of a module whose work always fuses with a sibling's: the
        structure is there, and nothing is attributed to it alone.

        Returns:
            Dictionary mapping module names to ModuleTiming objects at the top level

        """
        root_modules: dict[str, ModuleTiming] = {}

        def module_at(path: tuple[str, ...]) -> ModuleTiming | None:
            """Find or create the module at *path*, creating ancestors as needed."""
            level = root_modules
            module: ModuleTiming | None = None
            for frame in path:
                if frame not in level:
                    created = ModuleTiming(
                        name=frame, operation_timings=[], children=[]
                    )
                    level[frame] = created
                    if module is not None:
                        module.children.append(created)
                module = level[frame]
                level = {child.name: child for child in module.children}
            return module

        for timing in self.operation_timings:
            paths = [
                tuple(
                    _mlir.get_stack_trace(operation.location)  # type: ignore[attr-defined]
                    or ("<unknown>",)
                )
                for operation in timing.operations
            ]
            if not paths:
                continue

            # Every member's module exists in the tree, so the hierarchy is complete
            # whether or not a dispatch is filed against a given module.
            for path in paths:
                module_at(path)

            # The deepest module common to all members contains the whole dispatch.
            common = paths[0]
            for path in paths[1:]:
                limit = min(len(common), len(path))
                index = 0
                while index < limit and common[index] == path[index]:
                    index += 1
                common = common[:index]

            # Members spanning different roots are contained by no module.
            owner = module_at(common or ("<unknown>",))
            if owner is not None:
                owner.operation_timings.append(timing)

        return root_modules

    def write_summary(
        self: Self,
        output: TextIO,
        top_n: int | None = None,
        *,
        width: int | None = None,
    ) -> None:
        """
        Write benchmark results summary to output.

        One row per dispatch, since that is what the runtime measured: the duration
        belongs to the listed operations jointly, not to any one of them.

        Args:
            output: Text stream to write to
            top_n: If specified, only show top N slowest dispatches
            width: Console width to render at. Defaults to
                :data:`~coreai_torch.debugging.table_writer._DEFAULT_WIDTH`.

        """
        summary = self.get_operation_summary()
        if top_n is not None:
            summary = summary[:top_n]

        spec = _TableSpec(
            title="Benchmark Results",
            columns=(
                _Column("Core AI op ids"),
                _Column("Operations"),
                _Column("Median (ms)", justify="right"),
                _Column("Avg (ms)", justify="right"),
                _Column("Min (ms)", justify="right"),
                _Column("Max (ms)", justify="right"),
                _Column("StdDev (ms)", justify="right"),
            ),
            caption=(
                f"{len(self.operation_timings)} dispatches profiled. A row's duration "
                f"covers every operation listed in it."
            ),
            # Ids and names fold within their columns, so rule between rows to keep
            # it clear where a wide fused group ends.
            show_lines=True,
        )
        for timing in summary:
            spec.add(timing)

        _write_table(spec, output, width=width)


@dataclass(frozen=True)
class EventData:
    """Structured representation of a profiler log event's encoded data."""

    @dataclass(frozen=True)
    class Interval:
        """Timing interval information for an event."""

        duration: int
        """Duration of the interval (in nanoseconds)."""

        @staticmethod
        def from_dict(data: dict[str, Any]) -> "EventData.Interval":
            """
            Create an Interval instance from a parsed JSON dictionary.

            Args:
                data: Parsed JSON dictionary, e.g. ``{"duration": 1667}``

            Returns:
                Interval instance populated from the dictionary

            """
            return EventData.Interval(duration=data["duration"])

    interval: "EventData.Interval | None" = None
    """Interval timing information for the event, if present."""

    @dataclass(frozen=True)
    class CompileIdGroup:
        """The operations a fused dispatch was built from.

        The runtime emits one of these alongside the interval for a dispatch that
        fused several operations, keyed on the same compile identifiers, so the
        group's membership is stated rather than inferred.

        ``members`` are ids in the delegate's own numbering, not Core AI ids, so they
        need translating before they name anything in the converted program.
        """

        group_id: int
        """Identifier the runtime assigned this dispatch, equal to the event's
        ``delegate_id``. Assigned per dispatch, so it is meaningful only within the
        run that reported it."""

        members: tuple[int, ...]
        """Operation ids fused into this dispatch, in the order reported."""

        @staticmethod
        def from_dict(data: dict[str, Any]) -> "EventData.CompileIdGroup":
            """
            Create a CompileIdGroup from a parsed JSON dictionary.

            Args:
                data: Parsed JSON dictionary, e.g.
                    ``{"groupId": 4294967296, "members": [153, 154]}``

            Returns:
                CompileIdGroup instance populated from the dictionary

            """
            return EventData.CompileIdGroup(
                group_id=data["groupId"],
                members=tuple(data.get("members", ())),
            )

    compile_id_group: "EventData.CompileIdGroup | None" = None
    """Fused-dispatch membership for the event, if present."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EventData":
        """
        Create an EventData instance from a parsed JSON dictionary.

        The payload is a tagged union: an event carries an ``interval``, a
        ``compileIdGroup``, or neither.

        Args:
            data: Parsed JSON dictionary, e.g. ``{"interval": {"duration": 1667}}``
                or ``{"compileIdGroup": {"groupId": 1, "members": [2, 3]}}``

        Returns:
            EventData instance populated from the dictionary

        """
        interval_data = data.get("interval")
        group_data = data.get("compileIdGroup")
        return EventData(
            interval=(
                EventData.Interval.from_dict(interval_data) if interval_data else None
            ),
            compile_id_group=(
                EventData.CompileIdGroup.from_dict(group_data) if group_data else None
            ),
        )


def _event_payload(event: LogEvent) -> bytes | str | None:
    """
    The event's encoded payload as JSON, if this runtime exposes one.

    Only ``encoded_data`` carries JSON. The neighbouring ``data`` field is a
    rendering for people, so parsing it yields nothing and is not attempted; a
    runtime without ``encoded_data`` reports no payload at all.

    Args:
        event: LogEvent from the profiler.

    Returns:
        The payload, or ``None`` if the event exposes none.

    """
    return getattr(event, "encoded_data", None) or None


_GPU_INTERVAL_PREFIX = "GPU "
_HARDWARE_MARKER = " HW "


def _is_gpu_interval(event: LogEvent) -> bool:
    """
    Whether this event times one GPU encoder.

    Labelled ``GPU HW CB:0, Enc:3`` or ``GPU Host CB:0, Enc:3``. Activity codes, memory
    intervals and fused-op-group membership are not per-encoder measurements, so they
    must not be counted as timing this tool failed to recognise.

    Args:
        event: LogEvent from the profiler.

    Returns:
        True when the event measures a GPU encoder.

    """
    return event.event_id.startswith(_GPU_INTERVAL_PREFIX)


def _is_hardware_timestamped(event: LogEvent) -> bool:
    """
    Whether a GPU interval was measured by the GPU's own counters.

    Every encoder is reported twice, once from the GPU counters and once from the host
    clock. The host figure includes queueing and does not answer how long the operation
    took on the GPU, and both carry the same compile identifiers -- so recording both
    pools two different measurements into one sample list, where a median falls between
    them and a dispatch's samples run from near zero to the real duration.

    Args:
        event: LogEvent from the profiler.

    Returns:
        True when the interval came from the GPU counters.

    """
    return _HARDWARE_MARKER in event.event_id


def _parse_event_data(event_data: bytes | str | None) -> EventData:
    """
    Parse a log event's encoded data into a structured EventData object.

    The runtime hands this field over in more than one shape: JSON, as bytes or as
    str, and a placeholder such as ``"empty()"`` for an event carrying no payload.
    Anything unrecognised yields an EventData with no interval, so a caller simply
    finds nothing to attribute.

    Never raises. It runs inside a profiler callback invoked from a runtime thread,
    where an exception has nowhere to go and can take the process down with it.

    Args:
        event_data: The event's ``data`` field.

    Returns:
        EventData parsed from the payload, or an empty one if it holds no JSON
        object.

    """
    if not event_data:
        return EventData()

    if isinstance(event_data, bytes | bytearray):
        try:
            event_data = event_data.decode("utf-8")
        except UnicodeDecodeError:
            logger.debug("Profiler event data is not valid UTF-8; ignoring it")
            return EventData()

    try:
        parsed = json.loads(event_data)
    except (TypeError, ValueError):
        # Placeholders such as "empty()" are expected, not an error.
        logger.debug("Profiler event data is not JSON (%r); ignoring it", event_data)
        return EventData()

    if not isinstance(parsed, dict):
        return EventData()

    try:
        return EventData.from_dict(parsed)
    except (KeyError, TypeError):
        logger.debug("Profiler event data lacks an interval duration; ignoring it")
        return EventData()


class Benchmarker(ABC):
    """
    Base benchmarker class with common profiling logic.

    Subclasses implement runtime-specific execution methods.
    """

    def __init__(
        self: Self,
        coreai_program: AIProgram,
        entry_point: str = "main",
        excluded_operations: tuple[str, ...] | None = None,
        specialization_options: SpecializationOptions | None = None,
    ) -> None:
        """
        Initialize the base benchmarker.

        Args:
            coreai_program: AIProgram to benchmark
            entry_point: Name of the function to profile (default: "main")
            excluded_operations: Tuple of operation names to exclude from timing (default: None)
            specialization_options: Options for configuring model specialization

        """
        self.coreai_program = coreai_program
        self.entry_point = entry_point
        self.excluded_operations = excluded_operations or ()
        self.specialization_options = specialization_options
        self._intervals: dict[int, tuple[int, Any]] = {}
        # Keyed by ``(odix_id, sorted Core AI op ids)``, holding the dispatch's full
        # undivided duration once per run. Sorted because a delegate reports members
        # in a varying order, which otherwise splits one dispatch across entries.
        # ``delegate_id`` cannot key this: for a fused dispatch it is a per-dispatch
        # counter, so nothing would accumulate.
        self._timings: dict[tuple[int, tuple[int, ...]], list[float]] = defaultdict(
            list
        )
        self._interval_counter = 0
        self._debug_info_records: list[DebugInfoRecord] = []
        # Maps a compiled op identifier (odix_id, delegate_id) to the list of
        # coreai op IDs fused into it. Keyed on the same pair carried by a runtime
        # ``CompileIdentifiers`` object, so timing is attributed without collapsing
        # distinct delegate dispatches together.
        self._compile_id_to_coreai_map: dict[tuple[int, int | None], list[int]] = {}
        # Translates the operation ids a delegate reports into Core AI op ids, so a
        # dispatch it describes at run time can extend the map above.
        self._delegate_op_to_coreai_map: dict[int, list[int]] = {}
        # Durations whose compile identifier is not resolved yet: a delegate reports
        # a dispatch's timing before describing its membership.
        self._pending_durations: dict[tuple[int, int | None], list[float]] = (
            defaultdict(list)
        )
        self._coreai_operations: dict[int, Operation] = {}
        # A Condition rather than a Lock so the benchmark coroutine can wait for the
        # runtime's asynchronous callbacks to settle; otherwise used as a plain Lock.
        self._lock = threading.Condition()
        self._state = _BenchmarkerState.LOADING

    def _extract_coreai_operations(self: Self) -> None:
        """
        Extract Core AI operations from the coreai_program module.

        Walks the module and stores operations by their Core AI operation ID.
        """
        self._coreai_operations.clear()

        def walk_operations(op: Operation) -> WalkResult:
            op_id_obj = _mlir.get_operation_id(op.location, "coreai")  # type: ignore[attr-defined]
            if op_id_obj is not None:
                coreai_id = getattr(op_id_obj, "value", None)
                if coreai_id is not None:
                    self._coreai_operations[coreai_id] = op
            return WalkResult.ADVANCE

        self.coreai_program._mlir_module.operation.walk(walk_operations)

    def _build_compile_id_to_coreai_map(self: Self) -> None:
        """
        Build the mappings that turn compile identifiers into Core AI operation IDs.

        Two are needed, because timing arrives keyed two different ways:

        * ``(odix_id, delegate_id) -> [coreai ids]``, for identifiers the debug info
          describes statically. A single compiled op may fuse several Core AI ops, so
          each value is a list.
        * ``delegate op id -> [coreai ids]``, used to translate the membership a
          delegate reports for a dispatch whose identifier it assigned at run time.

        Called once after loading debug_infos.
        """
        self._compile_id_to_coreai_map = _build_compile_id_to_coreai_map(
            self._debug_info_records,
        )
        self._delegate_op_to_coreai_map = _build_delegate_op_to_source_map(
            self._debug_info_records,
        )

    def _get_coreai_op_ids(
        self: Self,
        odix_id: int,
        delegate_id: int | None,
    ) -> list[int]:
        """
        Convert compile identifiers to a list of Core AI operation IDs.

        Args:
            odix_id: ODIX ID from ``compile_ids.id``
            delegate_id: Delegate ID from ``compile_ids.delegate_id`` (may be None)

        Returns:
            List of Core AI operation IDs if found, otherwise empty list

        """
        return self._compile_id_to_coreai_map.get((odix_id, delegate_id), [])

    def _reset_state(self: Self) -> None:
        """Reset internal state before a new benchmark run."""
        with self._lock:
            self._intervals.clear()
            self._timings.clear()
            self._pending_durations.clear()
            self._interval_counter = 0

    def _wait_for_intervals_to_complete(self: Self, timeout_s: float = 10.0) -> None:
        """
        Block until every started interval has received its matching end event.

        Callbacks arrive asynchronously, so when ``await function(...)`` returns some
        intervals may be open. Reaching ``COMPLETED`` first makes their end events hit
        the ``state != RUNNING`` guard, dropping samples and varying the reported op
        set between runs. :meth:`_on_log_event_end` wakes this waiter when the last
        interval closes.

        Call while still ``RUNNING``, via :func:`asyncio.to_thread` so the event loop
        is not blocked.

        Args:
            timeout_s: Hard upper bound on the wait.

        """
        with self._lock:
            settled = self._lock.wait_for(
                lambda: not self._intervals,
                timeout=timeout_s,
            )
            if not settled:
                logger.warning(
                    "Timed out after %.1fs waiting for %d profiler interval(s) "
                    "to complete; results may be incomplete",
                    timeout_s,
                    len(self._intervals),
                )

    def _record_dispatch_duration(
        self: Self,
        compile_ids: Any,
        duration_ms: float,
    ) -> None:
        """
        Attribute one measured duration to the dispatch it belongs to.

        Recorded once for the whole dispatch, never divided across the operations it
        fused: a division reported a fraction of the real cost for each of them.
        An identifier the delegate assigned at run time is not in the static map and
        its membership is described only afterwards, so the duration waits in
        :attr:`_pending_durations` for :meth:`_resolve_compile_id_group` to flush it.

        Caller holds :attr:`_lock`.

        Args:
            compile_ids: The event's ``CompileIdentifiers``.
            duration_ms: Measured duration in milliseconds.

        """
        key = (compile_ids.id, compile_ids.delegate_id)
        coreai_ids = self._get_coreai_op_ids(*key)
        if coreai_ids:
            self._timings[(compile_ids.id, tuple(sorted(coreai_ids)))].append(
                duration_ms
            )
        else:
            self._pending_durations[key].append(duration_ms)

    def _resolve_compile_id_group(
        self: Self,
        compile_ids: Any,
        group: "EventData.CompileIdGroup",
    ) -> None:
        """
        Learn a dispatch's operations, and attribute whatever was waiting on it.

        Membership comes in the delegate's own numbering, so it is translated to
        Core AI ids and added to the map static attribution uses -- one place
        resolves a compile identifier, however it was described.

        Caller holds :attr:`_lock`.

        Args:
            compile_ids: The event's ``CompileIdentifiers``.
            group: Membership the delegate reported for this dispatch.

        """
        key = (compile_ids.id, compile_ids.delegate_id)

        coreai_ids: list[int] = []
        unresolved: set[int] = set()
        for member in group.members:
            mapped = self._delegate_op_to_coreai_map.get(member)
            if not mapped:
                unresolved.add(member)
                continue
            for coreai_id in mapped:
                if coreai_id not in coreai_ids:
                    coreai_ids.append(coreai_id)

        if unresolved:
            # Members absent from the debug info are attributed to nothing, so name
            # them rather than quietly reporting a smaller group. Which ids they are
            # is what says whether attribution is incomplete or the members simply
            # have no Core AI counterpart, as constants and memref views do not.
            logger.debug(
                "Dispatch (odix=%s, delegate=%s): %d of %d reported operations are "
                "absent from the debug info and are not attributed: %s",
                key[0],
                key[1],
                len(unresolved),
                len(group.members),
                sorted(unresolved),
            )

        pending = self._pending_durations.pop(key, [])
        if not coreai_ids:
            if pending:
                logger.warning(
                    "No Core AI operations resolved for compile id "
                    "(odix=%s, delegate=%s), dropping %d timing sample(s)",
                    key[0],
                    key[1],
                    len(pending),
                )
            return

        self._compile_id_to_coreai_map[key] = coreai_ids
        self._timings[(compile_ids.id, tuple(sorted(coreai_ids)))].extend(pending)

    def _on_log_event(self: Self, event: LogEvent) -> None:
        """
        Handle a self-contained profiler log event.

        Unlike the begin/end interval callbacks, this event carries its own payload:
        either a duration, attributed as in :meth:`_on_log_event_end`, or the
        membership of a fused dispatch, recorded in :attr:`compile_id_groups`.

        Args:
            event: LogEvent from the profiler

        """
        # Only process events when actively running benchmark
        if self._state != _BenchmarkerState.RUNNING:
            return

        # Only process inference phase events
        phase = _LogEventPhase(event.phase)
        if phase != _LogEventPhase.INFERENCE:
            return

        event_data = _parse_event_data(event_data=_event_payload(event))

        if event_data.compile_id_group is not None:
            with self._lock:
                self._resolve_compile_id_group(
                    event.compile_ids, event_data.compile_id_group
                )
            return

        # Without interval timing there is nothing to attribute.
        if event_data.interval is None:
            return

        # A GPU encoder first, then which clock measured it. Only the hardware
        # measurement answers how long the operation took on the GPU.
        if not (_is_gpu_interval(event) and _is_hardware_timestamped(event)):
            return

        with self._lock:
            self._record_dispatch_duration(
                event.compile_ids,
                float(event_data.interval.duration) / 1e6,
            )

    def _on_log_event_begin(self: Self, event: LogEvent) -> int:
        """
        Handle profiler interval begin events.

        Only processes events during RUNNING state and INFERENCE phase.

        Args:
            event: LogEvent from the profiler

        Returns:
            Interval ID for tracking this event

        """
        # Only process events when actively running benchmark
        if self._state != _BenchmarkerState.RUNNING:
            return 0  # Return dummy interval_id

        # Only process inference phase events
        phase = _LogEventPhase(event.phase)
        if phase != _LogEventPhase.INFERENCE:
            return 0  # Return dummy interval_id for non-inference events

        # Opening an interval for the host-timestamped twin would pool it with the
        # hardware measurement of the same encoder; its end event then finds no open
        # interval and is ignored.
        if _is_gpu_interval(event) and not _is_hardware_timestamped(event):
            return 0

        with self._lock:
            interval_id = self._interval_counter
            self._interval_counter += 1

            # Store interval start information
            self._intervals[interval_id] = (
                event.timestamp,
                event.compile_ids,
            )

            return interval_id

    def _on_log_event_end(self: Self, event: LogEvent, interval_id: int) -> None:
        """
        Handle profiler interval end events.

        Args:
            event: LogEvent from the profiler
            interval_id: Interval ID from the begin callback

        """
        # Only process events when actively running benchmark
        if self._state != _BenchmarkerState.RUNNING:
            return

        # Only process inference phase events
        phase = _LogEventPhase(event.phase)
        if phase != _LogEventPhase.INFERENCE:
            return

        with self._lock:
            # Retrieve start information (only INFERENCE events are stored)
            start_info = self._intervals.get(interval_id)
            if start_info is None:
                # Event was not stored or already processed
                return

            start_time, _ = start_info

            # Calculate duration in milliseconds and attribute it to the group
            duration_ns = event.timestamp - start_time
            self._record_dispatch_duration(event.compile_ids, float(duration_ns) / 1e6)

            # Waking the waiter once the last interval closes is what lets the
            # benchmark reach COMPLETED without dropping late samples.
            self._intervals.pop(interval_id)
            if not self._intervals:
                self._lock.notify_all()

    def _create_result(self: Self) -> BenchmarkResult:
        """
        Create BenchmarkResult from collected timings.

        Returns:
            BenchmarkResult containing timing information

        """
        with self._lock:
            # Convert raw timings to a list of OperationTiming, one per set of Core
            # AI ops measured together as a single dispatch.
            operation_timings_list = []
            for (_odix_id, group_ids), samples in self._timings.items():
                # Resolve each group member to its MLIR operation, dropping
                # excluded ops (e.g. coreai.graph / coreai.constant).
                group_ops: list[Operation] = []
                group_op_ids: list[int] = []
                for op_id in group_ids:
                    operation = self._coreai_operations.get(op_id)
                    if operation is None or operation.name in self.excluded_operations:
                        continue
                    group_ops.append(operation)
                    group_op_ids.append(op_id)

                # Skip groups whose members were all unknown or excluded.
                if not group_ops:
                    continue

                operation_timings_list.append(
                    OperationTiming(
                        op_ids=group_op_ids,
                        operations=group_ops,
                        measurement=Measurement.from_samples(samples),
                    )
                )

            # Durations whose description never arrived. Most measure a whole
            # function or delegate region -- a "symbol" in the debug info -- and are
            # excluded from per-op timing by design. The rest name no dispatch at
            # all, because an interval does not always carry a delegate_id.
            if self._pending_durations:
                symbol_odix_ids = {
                    operation.odix_id
                    for record in self._debug_info_records
                    for operation in record.operations
                    if operation.is_symbol()
                }
                symbols = {
                    compile_id: durations
                    for compile_id, durations in self._pending_durations.items()
                    if compile_id[0] in symbol_odix_ids
                }
                unnamed = {
                    compile_id: durations
                    for compile_id, durations in self._pending_durations.items()
                    if compile_id[0] not in symbol_odix_ids
                }
                if symbols:
                    logger.debug(
                        "%d sample(s) measure a whole function or delegate region "
                        "rather than an operation, and are excluded from per-op "
                        "timing: %s",
                        sum(len(d) for d in symbols.values()),
                        sorted(symbols, key=str),
                    )
                if unnamed:
                    logger.warning(
                        "%d timing sample(s) across %d compile identifier(s) were "
                        "measured but could not be attributed to any operation: %s",
                        sum(len(d) for d in unnamed.values()),
                        len(unnamed),
                        sorted(unnamed, key=str),
                    )

            return BenchmarkResult(
                operation_timings=operation_timings_list,
                operations_by_id=dict(self._coreai_operations),
            )

    @abstractmethod
    async def benchmark(
        self: Self,
        inputs: dict[str, Any],
        num_runs: int = 1,
    ) -> BenchmarkResult:
        """
        Benchmark the coreai_program program with the given inputs.

        Args:
            inputs: Dictionary mapping input names to tensor values
            num_runs: Number of timed iterations (default: 1). Always preceded by an
                untimed warmup run, which is not one of them, so the samples describe
                the steady state rather than first-inference costs.

        Returns:
            BenchmarkResult containing timing information for all operations

        """
        ...


class CoreAIBenchmarker(Benchmarker):
    """Benchmarker using Core AI Runtime."""

    async def benchmark(
        self: Self,
        inputs: dict[str, Any],
        num_runs: int = 1,
    ) -> BenchmarkResult:
        """
        Benchmark using Core AI Runtime.

        Args:
            inputs: Dictionary mapping input names to tensor values
            num_runs: Number of timed iterations (default: 1). Always preceded by an
                untimed warmup run, which is not one of them, so the samples describe
                the steady state rather than first-inference costs.

        Returns:
            BenchmarkResult containing timing information

        """
        # Reset state
        self._reset_state()

        # Create target (AIProgram) inspector based on inspector type
        with TemporaryDirectory() as temp_dir_name:
            asset_path = Path(temp_dir_name) / "model.aimodel"

            # Create asset from AIProgram and load model from asset
            asset = self.coreai_program.save_asset(asset_path)
            specialization_options = (
                self.specialization_options.with_debug(enabled=True)
                if self.specialization_options is not None
                else None
            )
            model = await AIModel.load(asset.path, specialization_options)
            # Load and parse debug_infos
            debug_infos_bytes = model._debug_infos
            self._debug_info_records = parse_debug_infos(debug_infos_bytes)

            # Extract Core AI operations from module
            self._extract_coreai_operations()

            # Build compile-id to Core AI ID mapping for fast lookup
            self._build_compile_id_to_coreai_map()

            # Create profiler with callbacks
            profiler = Profiler(
                on_log_event_begin=self._on_log_event_begin,
                on_log_event_end=self._on_log_event_end,
                on_log_event=self._on_log_event,
            )

            # Load function with profiler
            function = model.load_function(self.entry_point, profiler=profiler)
            if function is None:
                msg = f"Function '{self.entry_point}' not found in model"
                raise ValueError(msg)

            # Convert inputs to NDArray format
            nd_inputs = {}
            for name, value in inputs.items():
                if not isinstance(value, NDArray):
                    nd_inputs[name] = NDArray(value)
                else:
                    nd_inputs[name] = value

            # Run benchmark
            logger.info(
                "Running Core AI Runtime benchmark with %d iteration(s)...",
                num_runs,
            )

            # A first inference pays costs the steady state does not, and nothing
            # filters it out -- it dominated the mean, three orders of magnitude
            # above the median. Run one iteration before collection starts, so the
            # state guard discards its events.
            #
            # Unconditional, and not one of the `num_runs`: exempting `num_runs=1`
            # made the cheapest way to ask for a measurement the one way to get a
            # cold one, so a single-run number was not comparable with any other --
            # including the other side of a `timing_diff` comparison.
            logger.debug("Warmup run (untimed)")
            await function(nd_inputs)

            # Transition to RUNNING state to start collecting timing data
            self._state = _BenchmarkerState.RUNNING

            for i in range(num_runs):
                logger.debug("Benchmark run %d/%d", i + 1, num_runs)
                await function(nd_inputs)

            # Callbacks arrive asynchronously, so intervals from the final runs may
            # still be open. Block (off the event loop) until they close: flipping to
            # COMPLETED first drops late end events, and the reported op set then
            # varies between identical runs.
            await asyncio.to_thread(self._wait_for_intervals_to_complete)

            # All intervals have closed; stop collecting and build the result.
            self._state = _BenchmarkerState.COMPLETED

            result = self._create_result()
            logger.info(
                "Benchmark complete: %d operations profiled",
                len(result.operation_timings),
            )

            return result


async def benchmark_coreai_program(  # noqa: PLR0913
    coreai_program: AIProgram,
    inputs: dict[str, Any],
    entry_point: str = "main",
    num_runs: int = 1,
    excluded_operations: tuple[str, ...] | None = None,
    specialization_options: SpecializationOptions | None = None,
) -> BenchmarkResult:
    """
    Benchmark an AIProgram with profiling.

    Args:
        coreai_program: AIProgram to benchmark
        inputs: Dictionary mapping input names to tensor values
        entry_point: Name of the function to profile (default: "main")
        num_runs: Number of times to run the benchmark (default: 1)
        excluded_operations: Tuple of operation names to exclude from timing
                           (default: ("coreai.graph", "coreai.constant"))
        specialization_options: Options for configuring model specialization

    Returns:
        BenchmarkResult containing timing information for all operations

    """
    # Use default excluded operations if not specified
    if excluded_operations is None:
        excluded_operations = _get_default_excluded_operations()

    # Create the appropriate benchmarker implementation
    benchmarker = CoreAIBenchmarker(
        coreai_program, entry_point, excluded_operations, specialization_options
    )
    return await benchmarker.benchmark(inputs, num_runs)
