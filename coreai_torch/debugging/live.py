# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Core AI values on the tensors of an eager torch run.

Debugging a lowering usually means comparing two programs from the outside. This
puts the comparison where the author already is — inside ``forward``, at a
breakpoint — by giving every tensor the eager run produces a ``coreai``
attribute: the value the converted program computed for it, fetched from the
loaded ``.aimodel`` the first time it is read::

    from coreai_torch.debugging.live import live

    with live(model) as session:
        y = model(x)       # stop anywhere in forward: `h.coreai` is Core AI's h

    session.report()       # every tagged tensor, eager against Core AI

**The program is the caller's, built from the call being debugged.** ``live`` takes
a *build* — ``build(model, args, kwargs) -> (ExportedProgram, AIProgram)`` — and
runs it on the model's first call, with that call's arguments, before ``forward``
runs eagerly. Pass the pipeline the model ships through — its dynamic shapes, its
decompositions, its converter options — and the values are that program's; the
default exports, decomposes with ``get_decomp_table()`` and converts.

**The join is the export's own numbering.** ``torch.export`` traces under a
``TorchFunctionMode`` that counts every torch-level call per function and names
each node it produces ``{fn.__name__}_{count}`` — ``meta["torch_fn"]``. The eager
run here is traced by a mode that counts the same way, so the call that made a
tensor names the node that made it, without matching on files, lines or op
types. A call that lowers to several nodes — ``F.linear`` to a ``permute`` and an
``addmm`` — names the last of them, which is the value the call returned. From
the node, :func:`~.torch_utils.get_torch_to_coreai_output_mapping` gives the
Core AI op, and :class:`~.inspector.CoreAIInspector` its value.

The numbering restarts at each call of the root module, and the values come from
*that call's* inputs, so a session can follow several calls.

**One session per kind of call.** The names are call counts, so they line up only
while the eager run makes the export's calls in the export's order. Building from
the first call makes that hold for it by construction: the export traces the very
call that then runs, from the same Python state — export restores the module's
attributes after tracing. A later call is numbered against the same program, so
it must take the same path: a decode step after a prefill, where ``forward``
branches on Python state such as ``if past_key_values is None``, shifts every
later name onto another call's node, and those tensors carry *wrong* values with
nothing to say so. This is not detected; debug the other kind of call in a session
of its own. Branches the export resolves itself are safe: a static ``if`` or a
``torch.cond``, whose branches are subgraphs numbered on their own.

Read ``coreai`` from a debugger or after the call, not from code in ``forward``:
the export runs ``forward`` too, on fake tensors that carry no value, so code
there that reads one has to tolerate its absence — ``getattr(t, "coreai", None)``.

**Externalized composites are one call.** A build that externalizes submodules
-- ``export_and_convert(..., externalize_modules=specs)``, or a deployment
pipeline's own ``add_pytorch_module`` -- exports each as one call to a custom op.
The eager run counts the submodule's call as that one call and tags what it
returns, read from the export (the op in ``torch_fn``, the submodule in
``nn_module_stack``); the calls its body makes are behind it, untagged, as they
are in the export. The body still runs, and a debugger can step into it.

A composite's output -- ``layer_norm``, ``scaled_dot_product_attention``, an
externalized ``RMSNorm`` -- has a value only where Core AI maps the composite's
call through the interpreter's inlining of it (rdar://189232142); without that,
it is unobservable.

**A reference model splits an error in two.** ``live(model, build,
reference=float_model)`` runs a second eager model -- the float original of a
quantized ``model``, say -- on a copy of each call's arguments, and pairs the
two by module path. Each module output then carries ``reference``, the reference
model's output for that module, beside ``coreai``, and
:meth:`LiveSession.compare` measures both legs: the *model* error, eager against
reference, which is what quantization cost; and the *lowering* error, Core AI
against eager, which is what conversion cost. The pairing is by module, not by
call: quantization adds torch calls -- a dequantization per weight -- so the two
models' call counts differ, while their module trees agree, coreai-opt's
parametrizations only adding modules beside the ones it quantizes. A module
called more than once in a forward pairs call for call. The reference runs from
the same arguments, state included, so its error is per call: a decode step
compares from the quantized model's cache, not from one the reference kept.

Ported in design from ``coreai_nn.debugging.live``, which does the same for a
model authored over the layer library.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import operator
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Protocol

import coreai.runtime
import numpy as np
import torch
from coreai.runtime import AIModel, NDArray, SpecializationOptions
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten, tree_map

from ..converter import TorchConverter
from .inspector import CoreAIInspector
from .torch_utils import (
    _extract_output_mappings_from_coreai_program,
    _extract_torch_source_info_from_location,
    get_torch_node_names,
    get_torch_to_coreai_output_mapping,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from coreai.authoring import AIProgram
    from torch.export import ExportedProgram

    from .debug_info import OutputMapping

__all__ = [
    "ATTRIBUTE",
    "Build",
    "CoreAIValue",
    "LiveSession",
    "REFERENCE_ATTRIBUTE",
    "ReferenceValue",
    "Unobservable",
    "export_and_convert",
    "live",
]

ATTRIBUTE = "coreai"
"""The attribute each tagged tensor carries its :class:`CoreAIValue` under."""

REFERENCE_ATTRIBUTE = "reference"
"""The attribute a module's output carries its :class:`ReferenceValue` under."""


class Unobservable(LookupError):
    """The converted program computed this value somewhere it does not expose."""


class Build(Protocol):
    """
    Builds the program to debug from one call: ``build(model, args, kwargs)``.

    Returns the exported program and the Core AI program converted from it. The
    export must be the one the conversion took — decomposed already, if the
    conversion was — since the eager run is joined to its node names.
    """

    def __call__(
        self, model: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[ExportedProgram, AIProgram]: ...


def export_and_convert(
    model: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    externalize_modules: Sequence[Any] | None = None,
) -> tuple[ExportedProgram, AIProgram]:
    r"""
    The default build: export, decompose with ``get_decomp_table()``, convert.

    With *externalize_modules* -- ``ExternalizeSpec``\ s, as
    :meth:`~coreai_torch.TorchConverter.add_pytorch_module` takes them -- the
    matching submodules become composites, as a deployment pipeline that uses
    them converts them. Pass it with :func:`functools.partial`.
    """
    import coreai_torch  # noqa: PLC0415 — the package imports this module

    def export(module: torch.nn.Module) -> ExportedProgram:
        return torch.export.export(module, args, kwargs).run_decompositions(
            coreai_torch.get_decomp_table()
        )

    if not externalize_modules:
        exported_program = export(model)
        return exported_program, TorchConverter().add_exported_program(
            exported_program
        ).to_coreai()
    converter = TorchConverter().add_pytorch_module(
        model, export_fn=export, externalize_modules=list(externalize_modules)
    )
    program = converter.to_coreai()
    # The export the conversion took: the whole model, its composites as calls.
    return converter.exported_program, program


def _uses_os_framework() -> bool:
    """Which Core AI framework the runtime loaded, as the runtime recorded it at import."""
    return bool(getattr(coreai.runtime, "_use_os_coreai", False))


def _kind(dtype: Any) -> str:
    """Float, integer or bool: what must agree, where widths need not (int64 vs int32)."""
    if isinstance(dtype, torch.dtype):
        return "b" if dtype is torch.bool else ("f" if dtype.is_floating_point else "i")
    kind = np.dtype(dtype).kind
    return "i" if kind == "u" else kind


def _fits(array: np.ndarray, shape: Sequence[int], kind: str | None = None) -> bool:
    """Whether *array* can be a value of *shape*, and of element *kind* when given."""
    return tuple(array.shape) == tuple(shape) and (
        kind is None or _kind(array.dtype) == kind
    )


@dataclass(frozen=True)
class _Origin:
    """The node behind one ``torch_fn`` name, and where the converted program has it."""

    node: str
    mapping: OutputMapping | None
    reshaped: bool = False
    """Whether the op holds the value's data in another shape: an elided ``view``."""


def _results_of(call: str, nodes: list[torch.fx.Node]) -> dict[str, torch.fx.Node]:
    """
    The nodes carrying what one torch call returned, keyed as the eager side names them.

    A call adds a small subgraph — ``F.linear`` a ``permute`` and an ``addmm`` —
    and what it returned is that subgraph's *output*: the nodes whose value leaves
    it, used by something outside or by nothing at all. Not the last node in graph
    order, which agrees for a single result and is wrong for several: a
    ``torch.split`` adds one ``getitem`` per piece, and the last is only the last
    piece.

    A ``getitem`` output is keyed ``call[index]`` by its own index, so a piece the
    export dropped as dead does not shift the others. A single other output is
    keyed by the call. More than one other output names no single tensor, so none
    is keyed — an untagged tensor is a gap, a wrongly tagged one a wrong answer.
    """
    members = set(nodes)
    outputs = [
        n for n in nodes if not n.users or any(u not in members for u in n.users)
    ]
    pieces = {n: n.args[1] for n in outputs if n.target is operator.getitem}
    whole = [n for n in outputs if n not in pieces]
    results = {f"{call}[{index}]": node for node, index in pieces.items()}
    if len(outputs) == 1:
        # One output is the value, a `getitem` included: `native_layer_norm` is a
        # tuple at the aten level while `layer_norm` returns one tensor.
        results[call] = outputs[0]
    elif len(whole) > 1:
        results.clear()
    return results


_IDENTITIES = frozenset(
    {
        torch.ops.aten.clone.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.lift_fresh_copy.default,
    }
)
"""Nodes the converter elides, because their value is their input's."""

_VIEWS = frozenset(
    {
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.reshape.default,
    }
)
"""
Nodes whose value is their input's data in another shape, when the converter elides them.

A 3-D ``Linear`` decomposes to ``view → addmm → view`` and a batched ``matmul`` to
``view → bmm → view``; the last ``view`` holds the call's value and often has no
Core AI op of its own.
"""


def _source_names(program: Any) -> dict[int, str]:
    """Every torch node name the program can cite, by source op id."""
    return get_torch_node_names(program)


def _entry_names(program: Any, function_name: str) -> set[str]:
    """The torch node names the ops of *function_name*'s body were lowered from."""
    names: set[str] = set()

    def collect(operation: Any) -> None:
        info = _extract_torch_source_info_from_location(operation.location)
        if info is not None:
            names.add(info[1])
        for region in operation.regions:
            for block in region:
                for child in block:
                    collect(child)

    module = program._module._mlir_module  # noqa: SLF001
    for operation in module.body.operations:
        sym_name = (
            operation.attributes["sym_name"]
            if "sym_name" in operation.attributes
            else None
        )
        if sym_name is not None and str(sym_name).strip('"') == function_name:
            for region in operation.regions:
                for block in region:
                    for child in block:
                        collect(child)
    return names


def _check_pair(
    exported_program: ExportedProgram, program: Any, function_name: str
) -> None:
    """
    Refuse a build whose program was not converted from its exported program.

    The eager run is joined to the export's node names and through them to the
    program's ops, so a program converted from anything else — a re-export, an
    undecomposed export beside a decomposed conversion — joins to the wrong ops.
    Every node the entry function's ops name must be a node of the export. A
    composite's body is converted from an export of its own, and sits behind
    the call the export records; it is not part of the join.
    """
    nodes = {node.name for node in exported_program.graph.nodes}
    foreign = sorted(_entry_names(program, function_name) - nodes)
    if foreign:
        shown = ", ".join(foreign[:5]) + (", ..." if len(foreign) > 5 else "")
        msg = (
            f"build returned a program converted from another export: it names "
            f"{len(foreign)} node(s) the exported program has none of ({shown}). "
            f"Return the exported program the program was converted from, "
            f"after any run_decompositions"
        )
        raise ValueError(msg)


def _as_dtype(array: np.ndarray, dtype: str | None) -> np.ndarray:
    """
    *array* in the element type the program declares, when only the width differs.

    The converter narrows a 64-bit integer input to 32 bits, and the runtime
    refuses the eager call's ``int64``. Another kind -- a float for an integer --
    is left to the runtime to refuse, rather than converted into a different
    value.
    """
    try:
        target = np.dtype(dtype) if dtype else None
    except TypeError:
        return array
    if target is None or target == array.dtype or _kind(target) != _kind(array.dtype):
        return array
    return array.astype(target)


def _in_signature_order(
    model: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[Any]:
    """
    A call's arguments in ``forward``'s parameter order, as the export lists them.

    Keyword arguments are matched by name, so ``model(k_cache=k, input_ids=i)``
    lines up with an export whose inputs are ``input_ids, ..., k_cache``.
    """
    try:
        bound = inspect.signature(model.forward).bind(*args, **kwargs)
    except TypeError:
        return [args, kwargs]
    return list(bound.arguments.values())


def _argument_names(
    exported_program: ExportedProgram,
    input_names: Sequence[str],
    state_names: Sequence[str],
) -> list[str]:
    """
    The program's name for each of the export's user inputs, in order.

    An input ``forward`` updates in place -- a KV cache -- is the program's
    state, under the state's name; every other input is an input. Both keep the
    export's order.
    """
    signature = exported_program.graph_signature
    mutated = set(signature.user_inputs_to_mutate.values())
    inputs, states = iter(input_names), iter(state_names)
    names: list[str] = []
    for user_input in signature.user_inputs:
        name = next(states if user_input in mutated else inputs, None)
        if name is None:
            msg = (
                f"the program's inputs {list(input_names)} and states "
                f"{list(state_names)} do not match the export's user inputs "
                f"{list(signature.user_inputs)}: state held in a module buffer is "
                f"not supported"
            )
            raise ValueError(msg)
        names.append(name)
    return names


def _resolver(
    program: Any,
    computing: set[str],
) -> Callable[[torch.fx.Node], tuple[OutputMapping | None, bool]]:
    """
    Find the Core AI op output that holds a node's value.

    :func:`~.torch_utils.get_torch_to_coreai_output_mapping` answers for most nodes,
    and resolves one node lowered to several ops. It keeps one mapping per node,
    though, so a multi-output node — ``split_with_sizes``, ``max`` — keeps only one
    of its outputs. Its ``getitem`` pieces, which the converter elides, are found
    instead in the per-output maps, under the parent at the piece's index. An
    identity the converter elides has its input's value, so it is followed there.

    *computing* names the export's nodes that compute a value -- ``call_function``
    nodes other than views and identities -- and only they compete for what an op
    holds. A view or identity only moves its input's data, and a compiler that
    folds one into its neighbours can leave its output map on every op it was
    folded into.
    """
    resolved = get_torch_to_coreai_output_mapping(program)
    names = _source_names(program)
    per_output: dict[tuple[str, int], OutputMapping] = {}
    candidates: dict[str, list[OutputMapping]] = {}
    claimants: dict[tuple[int, int], set[str]] = {}
    for mapping in _extract_output_mappings_from_coreai_program(program):
        if mapping.source_level != "torch" or mapping.target_level != "coreai":
            continue
        name = names.get(mapping.source_op_id, "")
        key = (name, mapping.source_output)
        existing = per_output.get(key)
        # Several ops for one output: the last in dataflow order holds the value.
        if existing is None or mapping.target_op_id > existing.target_op_id:
            per_output[key] = mapping
        if mapping.source_output == 0:
            candidates.setdefault(name, []).append(mapping)
        if name in computing:
            # A parameter or input folded into an operand computes nothing, so it
            # does not compete for what the op holds.
            target = (mapping.target_op_id, mapping.target_output)
            claimants.setdefault(target, set()).add(name)

    def own_op(name: str) -> OutputMapping | None:
        """
        For a node the shared mapping dropped, the op that only it claims.

        The compiler can attribute one node to two ops: an MLP's last ``addmm``
        maps to its own bias add *and* to the residual add after it, which the
        residual's ``add`` also claims. :func:`get_torch_to_coreai_output_mapping`
        picks the residual add, finds it claimed by the ``add`` too, and drops the
        ``addmm`` — though its bias add holds its value. Only an op no other node
        claims is taken, and only when there is exactly one: anything else is a
        guess, which is how a wrong value gets in.
        """
        own = [
            m
            for m in candidates.get(name, ())
            if claimants.get((m.target_op_id, m.target_output)) == {name}
        ]
        return own[0] if len(own) == 1 else None

    def resolve(node: torch.fx.Node) -> tuple[OutputMapping | None, bool]:
        """The mapping, and whether the value is reached through an elided view."""
        if node.name in resolved:
            return resolved[node.name], False
        if node.target is not operator.getitem and (mapping := own_op(node.name)):
            return mapping, False
        if node.target is operator.getitem:
            parent, index = node.args
            return per_output.get((getattr(parent, "name", ""), index)), False
        source = node.args[0] if node.args else None
        if node.target in _IDENTITIES and isinstance(source, torch.fx.Node):
            return resolve(source)
        if node.target in _VIEWS and isinstance(source, torch.fx.Node):
            return resolve(source)[0], True
        return None, False

    return resolve


_EXTERNALIZED = "coreai_torch_ext."
"""Namespace of the custom ops an externalized submodule is exported as."""


def _composite_calls(exported_program: ExportedProgram) -> dict[str, str]:
    """
    The externalized submodules the export calls, by module path, with their op.

    An externalized submodule is exported as one call to a custom op, numbered
    like any torch call (``<op>.default_<n>``), so the eager run has to count
    the submodule's call as one call of that op too. Its node records both the
    op -- in ``torch_fn`` -- and the submodule that made the call -- the
    innermost ``nn_module_stack`` entry.
    """
    calls: dict[str, str] = {}
    for node in exported_program.graph.nodes:
        if node.op != "call_function" or not str(node.target).startswith(_EXTERNALIZED):
            continue
        torch_fn = node.meta.get("torch_fn")
        stack = node.meta.get("nn_module_stack") or {}
        if torch_fn is None or not stack:
            continue
        path = next(reversed(stack.values()))[0]
        calls[path] = torch_fn[0].rsplit("_", 1)[0]
    return calls


@dataclass(frozen=True)
class _Output:
    """A module's output: the tensor, and a copy of its value when the module returned it."""

    tensor: torch.Tensor
    snapshot: torch.Tensor


def _tensors(output: Any) -> list[torch.Tensor]:
    return [leaf for leaf in tree_flatten(output)[0] if isinstance(leaf, torch.Tensor)]


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    """``||value - reference|| / ||reference||``, in float64 -- a mask is boolean."""
    value64, reference64 = value.astype(np.float64), reference.astype(np.float64)
    scale = max(float(np.linalg.norm(reference64)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(value64 - reference64)) / scale


class ReferenceValue:
    """
    The reference model's output for the module call that returned this tensor.

    Hung on the output by :func:`live` when it is given a ``reference``, so a
    reader at a breakpoint anywhere -- in a file whose globals hold no session --
    gets the comparison from the tensor alone. Its ``repr`` is :meth:`compare`'s
    errors beside the value; reading it runs the program once, for the lowering
    error, as reading ``coreai`` does.
    """

    __slots__ = ("_session", "_value", "call", "module", "output")

    def __init__(
        self,
        session: LiveSession,
        module: str,
        call: int,
        output: int,
        value: torch.Tensor,
    ) -> None:
        self._session = session
        self.module = module
        """The module path, in both models."""
        self.call = call
        """Which of the module's calls in the forward this is, from zero."""
        self.output = output
        """Which tensor of the module's output this is."""
        self._value = value

    @property
    def tensor(self) -> torch.Tensor:
        """The reference's output, as a tensor."""
        return self._value

    def numpy(self) -> np.ndarray:
        """The reference's output."""
        return self._value.numpy()

    def compare(self) -> dict[str, Any]:
        """This output against the reference and against Core AI: :meth:`LiveSession.compare`."""
        return self._session.compare(self.module, call=self.call, output=self.output)

    def __repr__(self) -> str:
        """The module, the model and lowering errors, and the reference's value."""
        with self._session.paused():
            row = self.compare()

            def error(name: str) -> str:
                value = row[name]
                return "unmeasured" if value is None else f"{value:.2e}"

            note = f"; {row['note']}" if row["note"] else ""
            name = self.module or "<model>"
            return (
                f"ReferenceValue({name} call {self.call}: model error "
                f"{error('model_error')}, lowering error {error('lowering_error')}"
                f"{note})\n"
                f"{np.array2string(self.numpy(), precision=4, suppress_small=True)}"
            )


class CoreAIValue:
    """
    What the converted program computed for one eager tensor.

    Fetched on first read and cached for the call, so a debugger's Variables pane
    pays for a value once. Its ``repr`` shows the value beside the eager one's
    difference, which is what a reader at a breakpoint wants to know first.
    """

    __slots__ = ("_eager", "_session", "key")

    def __init__(self, session: LiveSession, key: str, eager: torch.Tensor) -> None:
        self._session = session
        self.key = key
        # The value at the moment the call returned it: an in-place op later in
        # forward would otherwise change what this compares against.
        self._eager = eager.detach().clone()

    @property
    def node(self) -> str:
        """The exported node this tensor's call produced last."""
        return self._session.origin(self.key).node

    def numpy(self) -> np.ndarray:
        """The Core AI value. Raises :class:`Unobservable` when the program hides it."""
        return self._session.fetch(
            self.key, tuple(self._eager.shape), _kind(self._eager.dtype)
        )

    @property
    def eager(self) -> np.ndarray:
        """The eager value it is compared against."""
        return self._eager.numpy()

    @property
    def max_abs_diff(self) -> float:
        """The largest absolute difference from the eager value."""
        return float(
            np.max(
                np.abs(self.numpy().astype(np.float64) - self.eager.astype(np.float64))
            )
        )

    def __repr__(self) -> str:
        """The node, the difference from eager, and the value — or why there is none."""
        with self._session.paused():
            try:
                value = self.numpy()
            except Unobservable as missing:
                return (
                    f"CoreAIValue({self.key} -> {self.node}: unobservable — {missing})"
                )
            delta = self.max_abs_diff
            return (
                f"CoreAIValue({self.key} -> {self.node}, max|Δ| vs eager {delta:.1e})\n"
                f"{np.array2string(value, precision=4, suppress_small=True)}"
            )


_DEBUGGERS = ("pdb", "bdb", "cmd", "ipdb", "pydevd", "_pydevd", "_pydev", "debugpy")
"""Top-level packages whose code evaluates expressions on a paused program."""

_DEBUGGER_MODULES = ("IPython.core.debugger", "IPython.terminal.debugger")
"""Debuggers inside packages that are otherwise not one — IPython runs cells too."""

_NOT_A_CALLER = ("torch", __name__)
"""Code between a torch call and whoever made it: torch's dispatch, and this module."""


def _is_debugger(frame: Any) -> bool:
    """Whether *frame* runs a debugger's own code."""
    module = frame.f_globals.get("__name__", "") if frame is not None else ""
    return module.split(".")[0] in _DEBUGGERS or module.startswith(_DEBUGGER_MODULES)


def _from_a_debugger() -> bool:
    """
    Whether the torch call being handled was made by a debugger, not by ``forward``.

    At a breakpoint the debugger makes torch calls of its own — ``p torch.abs(h)``
    at a ``pdb`` prompt, a watch expression, an IDE's Variables pane — and counting
    them would shift every later call of that function onto the wrong node. They
    are recognised by who made them: the first frame outside torch and this
    module is the caller, and either it is the debugger's code (a Variables pane
    reading a tensor) or the debugger is what called it (``pdb``'s ``p`` evaluates
    a string, whose frame borrows the paused frame's globals but is entered from
    ``pdb``).
    """
    frame = sys._getframe(2)  # noqa: SLF001 — past __torch_function__ and this function
    while frame is not None and frame.f_globals.get("__name__", "").startswith(
        _NOT_A_CALLER
    ):
        frame = frame.f_back
    return frame is not None and (_is_debugger(frame) or _is_debugger(frame.f_back))


class _Tagging(TorchFunctionMode):
    """
    Counts torch-level calls the way export does, and tags what they return.

    Export's ``TorchFunctionMetadataMode`` increments a counter per function
    object on every ``__torch_function__`` it sees, and names a node after the
    current count. Doing exactly that here is what makes the names agree. Nested
    calls are invisible to both: a mode is off while its own handler runs.
    """

    def __init__(self, session: LiveSession) -> None:
        super().__init__()
        self.session = session
        self.counts: dict[Any, int] = {}
        self.pause_depth = 0

    def __torch_function__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if torch.compiler.is_compiling():
            # `torch.cond` and its kin compile their branches with Dynamo, which
            # traces this handler too; the frame inspection below would break that
            # graph and `cond` refuses to run. A compiled branch is a subgraph of
            # the export with its own counter, so its calls are not ours to number.
            return func(*args, **(kwargs or {}))
        result = func(*args, **(kwargs or {}))
        if self.pause_depth or _from_a_debugger():
            return result
        count = self.counts[func] = self.counts.get(func, 0) + 1
        name = getattr(func, "__name__", None)
        if name is None:
            return result
        self.tag_result(f"{name}_{count}", result)
        return result

    def tag_result(self, call: str, result: Any) -> None:
        """Tag what the call named *call* returned: one tensor, or each of several."""
        if isinstance(result, torch.Tensor):
            if call in self.session.origins:
                self.session.tag(result, call)
        elif isinstance(result, (tuple, list)):
            # `torch.max(dim=)` and `torch.split` return several tensors; the
            # export gives each its own `getitem`, keyed by position.
            for position, item in enumerate(result):
                key = f"{call}[{position}]"
                if isinstance(item, torch.Tensor) and key in self.session.origins:
                    self.session.tag(item, key)

    def enter_composite(self) -> None:
        """A composite's body runs: its calls are behind the one the export records."""
        self.pause_depth += 1

    def leave_composite(self, op: str, result: Any) -> None:
        """
        Count the composite as one call of its op, and tag what it returned.

        A composite inside another is behind the outer one's call, as in the
        export, so only the outermost is counted.
        """
        self.pause_depth -= 1
        if self.pause_depth:
            return
        count = self.counts[op] = self.counts.get(op, 0) + 1
        self.tag_result(f"{op}_{count}", result)


class LiveSession:
    """
    A converted program held open beside an eager model, answering one value per read.

    Built by :func:`live`. The program is built on the model's first call, from
    that call's arguments, and loaded on its own event loop in a background
    thread, so a read can happen anywhere — inside ``forward``, at a ``pdb``
    prompt, in an IDE's Variables pane, or in a notebook whose loop is already
    running — without the caller being async.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        build: Build | None = None,
        *,
        specialization_options: SpecializationOptions | None = None,
        function_name: str = "main",
        reference: torch.nn.Module | None = None,
    ) -> None:
        """
        Hold *model* and how to build its program; nothing is built until it is called.

        Args:
            model: The eager module to debug.
            build: ``build(model, args, kwargs) -> (exported_program, program)``,
                run on the first call's arguments. Defaults to
                :func:`export_and_convert`.
            specialization_options: Where to load. ``None`` is the interpreter,
                which fuses nothing, so every call has a value; it needs the
                bundled runtime, since the OS framework's interpreter reports no
                intermediates. Debug is enabled on top of anything else.
            function_name: The program's function the build exported — ``"main"``
                unless it converts several entry points.
            reference: An eager model to compare against, module by module: the
                float original of a quantized *model*. Every module path it has
                must exist in *model*; *model* may have more.

        """
        if build is not None and not callable(build):
            msg = (
                f"build must be a function, got {type(build).__name__}: the program "
                f"is built from the model's first call, so no example inputs are "
                f"passed — use live(model), or live(model, build)"
            )
            raise TypeError(msg)
        self.model = model
        self.function_name = function_name
        self._build_program = build or export_and_convert
        self._options = _resolve(specialization_options)
        self._building = False
        self.exported_program: ExportedProgram | None = None
        """The build's export, once the first call has built it."""
        self.origins: dict[str, _Origin] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

        self._mode = _Tagging(self)
        self._composite_hooks: list[Any] = []
        self._inputs: dict[str, np.ndarray] = {}
        self._cache: dict[str, np.ndarray] = {}
        self.values: list[CoreAIValue] = []
        """Every tensor tagged during the latest call of the model, in call order."""

        if reference is not None:
            ours = dict(model.named_modules())
            missing = [
                name for name, _ in reference.named_modules() if name not in ours
            ]
            if missing:
                msg = (
                    f"the reference has modules the model does not: {missing[:5]}"
                    f"{' ...' if len(missing) > 5 else ''}. Its outputs are paired with "
                    f"the model's by module path, so every path must exist in both"
                )
                raise ValueError(msg)
        self.reference = reference
        """The reference model, if any."""
        self._reference_paths = (
            [name for name, _ in reference.named_modules()] if reference else []
        )
        self._reference_hooks: list[Any] = []
        self._module_outputs: dict[str, list[list[_Output]]] = {}
        self._reference_outputs: dict[str, list[list[torch.Tensor]]] = {}

    def _build(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Build the program from this call, join the export to it, and load it."""
        # The export runs forward too, and calls the model — so this hook again,
        # which must not rebuild — on fake tensors: untagged, and quietly, so the
        # author's breakpoints stop only in the eager run, where tensors have values.
        self._building = True
        try:
            with _without_modes(), _quiet():
                exported_program, program = self._build_program(
                    self.model, args, kwargs
                )
        finally:
            self._building = False
        _check_pair(exported_program, program, self.function_name)
        self.exported_program = exported_program
        self._hook_composites(_composite_calls(exported_program))

        resolve = _resolver(
            program,
            {
                n.name
                for n in exported_program.graph.nodes
                if n.op == "call_function" and n.target not in _VIEWS | _IDENTITIES
            },
        )
        groups: dict[str, list[torch.fx.Node]] = {}
        for node in exported_program.graph.nodes:
            torch_fn = node.meta.get("torch_fn")
            if node.op == "call_function" and torch_fn is not None:
                groups.setdefault(torch_fn[0], []).append(node)
        for call, nodes in groups.items():
            for key, node in _results_of(call, nodes).items():
                self.origins[key] = _Origin(node.name, *resolve(node))
        # The output node lists a stateful program's state updates before its
        # results; only the results are the program's outputs.
        self._result_names = list(exported_program.graph_signature.user_outputs)

        self._directory = TemporaryDirectory(prefix="coreai-torch-live-")
        asset = program.save_asset(Path(self._directory.name) / "model.aimodel")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        # An IDE debugger paused at a breakpoint suspends every thread it traces,
        # and its Variables pane then reads `coreai` on each tensor — which waits
        # on this thread. pydevd (debugpy) leaves a thread marked so running.
        self._thread.pydev_do_not_trace = True  # type: ignore[attr-defined]
        self._thread.start()
        self._model = self._call(AIModel.load(asset.path, self._options))
        self._inspector = CoreAIInspector(
            model=self._model, function_name=self.function_name
        )
        description = self._model.load_function(self.function_name).desc
        self._input_names = list(description.input_names)
        self._state_names = list(description.state_names)
        self._output_names = list(description.output_names)
        self._argument_names = _argument_names(
            exported_program, self._input_names, self._state_names
        )
        self._argument_dtypes = {
            **{n: description.input_descriptor(n).dtype for n in self._input_names},
            **{n: description.state_descriptor(n).dtype for n in self._state_names},
        }

    def _hook_composites(self, calls: dict[str, str]) -> None:
        """
        Count each externalized submodule's call as the one call the export made.

        The export ran the submodule as a custom op, and the eager run executes
        its body; uncounted, the body's calls would advance those functions'
        counts past the export's and misname every later call. The body runs
        normally -- a debugger can step into it -- but its tensors are behind
        the call, so they carry no value.
        """
        for path, op in calls.items():
            try:
                submodule = self.model.get_submodule(path)
            except AttributeError:
                continue

            def enter(_module: Any, _args: Any) -> None:
                self._mode.enter_composite()

            def leave(_module: Any, _args: Any, result: Any, _op: str = op) -> None:
                self._mode.leave_composite(_op, result)

            self._composite_hooks += [
                submodule.register_forward_pre_hook(enter),
                submodule.register_forward_hook(leave, always_call=True),
            ]

    def _unhook_composites(self) -> None:
        for handle in self._composite_hooks:
            handle.remove()
        self._composite_hooks = []

    # -- the eager side -------------------------------------------------------

    def tag(self, tensor: torch.Tensor, key: str) -> None:
        """Hang *key*'s :class:`CoreAIValue` on *tensor*."""
        with self.paused():
            value = CoreAIValue(self, key, tensor)
        try:
            setattr(tensor, ATTRIBUTE, value)
        except (AttributeError, TypeError):
            return
        self.values.append(value)

    @contextmanager
    def paused(self) -> Generator[None, None, None]:
        """
        Stop counting for the duration, so this module's own torch calls don't.

        A ``detach`` here, or a ``numpy`` in a ``repr``, would otherwise advance
        those functions' counts past the export's and misname every later call.
        """
        self._mode.pause_depth += 1
        try:
            yield
        finally:
            self._mode.pause_depth -= 1

    def _on_call(
        self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Start a call: build on the first, number from zero, fetch from its inputs."""
        if self._building:
            return
        if self._loop is None:
            self._build(args, kwargs)
        with self.paused():
            # Copied, not viewed: forward is about to update a state -- a KV
            # cache -- in place, and every run of the program steps from the
            # state as it was before this call.
            tensors = [
                np.array(leaf.detach().cpu().numpy(), copy=True)
                for leaf in tree_flatten(_in_signature_order(self.model, args, kwargs))[
                    0
                ]
                if isinstance(leaf, torch.Tensor)
            ]
        if len(tensors) != len(self._argument_names):
            msg = (
                f"the program takes {len(self._argument_names)} input(s) and state(s) "
                f"{self._argument_names}, but this call passed {len(tensors)} tensor(s)"
            )
            raise ValueError(msg)
        self._mode.counts.clear()
        self._cache.clear()
        self.values = []
        self._inputs = {
            name: _as_dtype(array, self._argument_dtypes.get(name))
            for name, array in zip(self._argument_names, tensors, strict=True)
        }
        self._module_outputs.clear()
        self._reference_outputs.clear()
        if self.reference is not None:
            self._run_reference(args, kwargs)

    # -- the reference model ----------------------------------------------------

    def _hook_reference(self) -> None:
        """Keep every module's output, in the model and in the reference, by path."""
        for path in self._reference_paths:
            self._reference_hooks += [
                self.reference.get_submodule(path).register_forward_hook(
                    functools.partial(self._keep_reference, path)
                ),
                self.model.get_submodule(path).register_forward_hook(
                    functools.partial(self._keep_output, path)
                ),
            ]

    def _unhook_reference(self) -> None:
        for hook in self._reference_hooks:
            hook.remove()
        self._reference_hooks = []

    def _run_reference(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """
        Run the reference on a copy of this call's arguments.

        Copied, because a forward updates state in place and the model has yet
        to run from it. Run as the export is -- from a clean mode stack, so
        nothing is counted, and quietly, since the two models usually share
        classes: a breakpoint in ``forward`` would otherwise stop in the
        reference first, on tensors with no ``coreai``.
        """
        with _without_modes(), _quiet(), torch.no_grad():
            copied_args, copied_kwargs = tree_map(
                lambda leaf: (
                    leaf.detach().clone() if isinstance(leaf, torch.Tensor) else leaf
                ),
                (args, kwargs),
            )
            self.reference(*copied_args, **copied_kwargs)

    def _keep_reference(self, path: str, _module: Any, _args: Any, output: Any) -> None:
        self._reference_outputs.setdefault(path, []).append(
            [tensor.detach().clone() for tensor in _tensors(output)]
        )

    def _keep_output(self, path: str, _module: Any, _args: Any, output: Any) -> None:
        """Keep a module's output, and hang the reference's on it, call for call."""
        if self._building:
            return
        with self.paused():
            tensors = _tensors(output)
            calls = self._module_outputs.setdefault(path, [])
            references = self._reference_outputs.get(path, [])
            if len(calls) < len(references):
                for output, (tensor, theirs) in enumerate(
                    zip(tensors, references[len(calls)], strict=False)
                ):
                    value = ReferenceValue(self, path, len(calls), output, theirs)
                    try:
                        setattr(tensor, REFERENCE_ATTRIBUTE, value)
                    except (AttributeError, TypeError):
                        continue
            # The tensor, for the `coreai` a later hook may still tag it with;
            # a copy, for the value it had when the module returned it.
            calls.append([_Output(t, t.detach().clone()) for t in tensors])

    def _path_of(self, module: torch.nn.Module) -> str:
        for model in (self.model, self.reference):
            for path, candidate in model.named_modules() if model is not None else ():
                if candidate is module:
                    return path
        msg = f"{type(module).__name__} is not a module of the model or its reference"
        raise LookupError(msg)

    def compare(
        self, module: torch.nn.Module | str, *, call: int = -1, output: int = 0
    ) -> dict[str, Any]:
        """
        One module's output, against the reference and against Core AI.

        Args:
            module: The module, in the model or the reference, or its path.
            call: Which of its calls in the latest forward, for a module called
                more than once; the last by default.
            output: Which tensor of its output, for a module returning several.

        Returns:
            The module, its shape, and three relative errors, ``||a - b|| / ||b||``:
            ``model_error``, eager against the reference -- what quantization
            cost; ``lowering_error``, Core AI against eager -- what conversion
            cost; and ``total_error``, Core AI against the reference. An error
            that cannot be measured is ``None``, and ``note`` says why.

        """
        if self.reference is None:
            msg = "compare() needs a reference model: live(model, build, reference=...)"
            raise ValueError(msg)
        path = module if isinstance(module, str) else self._path_of(module)
        with self.paused():
            return self._comparison(path, call, output)

    def _comparison(self, path: str, call: int, output: int) -> dict[str, Any]:
        calls = self._module_outputs.get(path, [])
        if not calls:
            msg = f"module {path!r} has not returned during the latest call"
            raise LookupError(msg)
        index = range(len(calls))[call]
        ours = calls[index][output]
        eager = ours.snapshot.numpy()
        row: dict[str, Any] = {
            "module": path,
            "call": index,
            "shape": tuple(eager.shape),
            "model_error": None,
            "lowering_error": None,
            "total_error": None,
            "note": "",
        }
        notes = []
        references = self._reference_outputs.get(path, [])
        theirs = (
            references[index][output].numpy()
            if index < len(references) and output < len(references[index])
            else None
        )
        if theirs is None:
            notes.append("the reference made no such call")
        elif theirs.shape != eager.shape:
            notes.append(f"the reference's output is {theirs.shape}")
            theirs = None
        else:
            row["model_error"] = _relative_error(eager, theirs)
        value = getattr(ours.tensor, ATTRIBUTE, None)
        if value is None:
            notes.append(
                "no torch call returned this output, so it has no Core AI value"
            )
        else:
            try:
                coreai_value = value.numpy()
            except Unobservable as missing:
                notes.append(str(missing))
            else:
                row["lowering_error"] = _relative_error(coreai_value, eager)
                if theirs is not None:
                    row["total_error"] = _relative_error(coreai_value, theirs)
        row["note"] = "; ".join(notes)
        return row

    def report_modules(self) -> list[dict[str, Any]]:
        """
        :meth:`compare` for every module output of the latest call, in module order.

        Each observable output costs one run of the program; a module called more
        than once has a row per call.
        """
        rows = []
        with self.paused():
            for path in self._reference_paths:
                for index, outputs in enumerate(self._module_outputs.get(path, [])):
                    rows += [
                        self._comparison(path, index, output)
                        for output in range(len(outputs))
                    ]
        return rows

    def report_weights(self) -> list[dict[str, Any]]:
        """
        Every reference parameter against the model's, as the model computes it.

        The model's is read as its module's attribute, so a parametrized weight --
        coreai-opt's quantized ``Linear`` -- is its dequantized value. A weight tied
        to another is one row.

        Returns:
            One row per parameter: its name, shape, the relative error
            ``||model - reference|| / ||reference||``, and the largest absolute
            difference.

        """
        if self.reference is None:
            msg = "report_weights() needs a reference model: live(model, build, reference=...)"
            raise ValueError(msg)
        rows = []
        with _without_modes(), torch.no_grad():
            for name, theirs in self.reference.named_parameters():
                path, _, attribute = name.rpartition(".")
                ours = getattr(self.model.get_submodule(path), attribute)
                ours_np, theirs_np = ours.detach().numpy(), theirs.detach().numpy()
                rows.append(
                    {
                        "weight": name,
                        "shape": tuple(theirs.shape),
                        "relative_error": _relative_error(ours_np, theirs_np),
                        "max_abs_diff": float(
                            np.max(
                                np.abs(
                                    ours_np.astype(np.float64)
                                    - theirs_np.astype(np.float64)
                                )
                            )
                        ),
                    }
                )
        return rows

    # -- the Core AI side -----------------------------------------------------

    def origin(self, key: str) -> _Origin:
        """The node and Core AI op behind a ``torch_fn`` name."""
        return self.origins[key]

    def fetch(
        self, key: str, shape: Sequence[int], kind: str | None = None
    ) -> np.ndarray:
        """
        The Core AI value for the call named *key*: a *shape* array of element *kind*.

        A capture is never trusted by position alone. The debug info can map an op
        to an instruction whose output is something else, so a capture that does
        not fit is discarded. And the bundled interpreter delivers a multi-output
        instruction's outputs in no fixed order — a ``split``'s two pieces arrive
        as ``[c, d]`` in one run and ``[d, c]`` in the next, against debug info
        that never changes — so among several outputs the one that fits is taken,
        and when more than one fits, the value is unobservable rather than a guess.
        A value that leaves the program as a result is read from a plain run,
        which names its outputs.
        """
        if key in self._cache:
            return self._cache[key]
        origin = self.origins[key]
        if origin.mapping is None:
            msg = f"node {origin.node} has no Core AI op: the conversion maps nothing to it"
            raise Unobservable(msg)
        op_id, output = origin.mapping.target_op_id, origin.mapping.target_output
        with self.paused():
            captured = self._call(
                self._inspector.get_intermediates_for_ops([op_id], self._inputs)
            )
        values = [v for v in (captured.get(op_id) or []) if v is not None]
        if origin.reshaped:
            # An elided view: the op holds this value's data in another shape.
            size = int(np.prod(shape))
            values = [
                v.reshape(shape)
                for v in values
                if v.size == size and (kind is None or _kind(v.dtype) == kind)
            ]
        fitting = [v for v in values if _fits(v, shape, kind)]
        ambiguous = len(fitting) > 1 and not all(
            np.array_equal(fitting[0], other) for other in fitting[1:]
        )
        if fitting and not ambiguous:
            self._cache[key] = np.array(fitting[0], copy=True)
            return self._cache[key]
        if origin.node in self._result_names:
            ran = self._call(self._run())
            position = self._result_names.index(origin.node)
            self._cache[key] = ran[self._output_names[position]]
            return self._cache[key]
        if ambiguous:
            msg = (
                f"coreai {op_id}[{output}]: {len(fitting)} of the op's outputs fit a "
                f"{tuple(shape)} value, and the runtime delivers them in no fixed order, "
                f"so which one this is cannot be told"
            )
            raise Unobservable(msg)
        if values:
            got = ", ".join(f"{tuple(v.shape)} {v.dtype}" for v in values)
            msg = (
                f"coreai {op_id}: the runtime returned {got} for a {tuple(shape)} value, "
                f"so its debug info points at another output"
            )
            raise Unobservable(msg)
        raise Unobservable(self._inspector.explain_missing(op_id).describe())

    async def _run(self) -> dict[str, np.ndarray]:
        """Run once, plainly, from this call's inputs and state, and copy every result out."""
        function = self._model.load_function(self.function_name)
        # Wrapping an array copies it, so the run's state update stays in the run.
        outputs = await function(
            {
                k: NDArray(v)
                for k, v in self._inputs.items()
                if k not in self._state_names
            },
            state={
                k: NDArray(v) for k, v in self._inputs.items() if k in self._state_names
            }
            or None,
        )
        return {
            name: np.array(outputs[name].numpy(), copy=True)
            for name in self._output_names
        }

    def _call(self, coroutine: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    # -- reporting ------------------------------------------------------------

    def report(self, *, rtol: float = 1e-4, atol: float = 1e-5) -> list[dict[str, Any]]:
        """
        Every tensor tagged in the latest call, eager against Core AI.

        Returns:
            One row per tensor, in call order: the ``torch_fn`` name, the node,
            the shape, the largest absolute difference, and whether the two agree
            within tolerance — or why there is no Core AI value.

        """
        rows: list[dict[str, Any]] = []
        with self.paused():
            for value in self.values:
                row: dict[str, Any] = {
                    "call": value.key,
                    "node": value.node,
                    "shape": tuple(value.eager.shape),
                }
                try:
                    # In float64: a boolean value -- a mask -- cannot be subtracted.
                    theirs = value.numpy().astype(np.float64)
                    eager = value.eager.astype(np.float64)
                except Unobservable as missing:
                    row |= {"max_abs_diff": None, "match": None, "note": str(missing)}
                else:
                    row |= {
                        "max_abs_diff": float(np.max(np.abs(theirs - eager))),
                        "match": bool(np.allclose(theirs, eager, rtol=rtol, atol=atol)),
                        "note": "",
                    }
                rows.append(row)
        return rows

    def close(self) -> None:
        """Release the loaded program and its background loop, if a call built one."""
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._directory.cleanup()


@contextmanager
def _without_modes() -> Generator[None, None, None]:
    """
    Set every torch function mode aside for the duration, and restore them in order.

    The build runs from inside the model's first call: under the session's
    tagging mode, and under any mode the model's own hooks entered for that call
    -- coreai-opt's eager quantization enters one around every forward. The
    export has to run as it would on its own, from a clean stack: it pushes its
    own counting mode, and the model's hooks enter theirs again for the export's
    call and require them on top when they leave.
    """
    set_aside = []
    try:
        while torch.overrides._get_current_function_mode_stack():  # noqa: SLF001
            set_aside.append(torch.overrides._pop_mode())  # noqa: SLF001
        yield
    finally:
        for mode in reversed(set_aside):
            torch.overrides._push_mode(mode)  # noqa: SLF001


@contextmanager
def _quiet() -> Generator[None, None, None]:
    """
    Run the export with ``forward``'s breakpoints and prints switched off.

    ``torch.export`` runs ``forward`` on fake tensors, which carry no value; a
    stop there would show the author a tensor without ``coreai``. The trace
    function is suspended — that is what an IDE's line breakpoints and
    ``python -m pdb``'s ``b file:line`` stop through — and so are ``breakpoint()``
    and ``pdb.set_trace()``, which bypass it. ``print`` in ``forward`` is muted
    too, so each prints once, in the eager run.
    """
    import io  # noqa: PLC0415
    import pdb  # noqa: PLC0415
    from contextlib import redirect_stdout  # noqa: PLC0415

    hook, set_trace, tracer = sys.breakpointhook, pdb.set_trace, sys.gettrace()
    settrace = _settrace()
    sys.breakpointhook = lambda *_, **__: None
    pdb.set_trace = lambda *_, **__: None
    if settrace is not None:
        settrace(None)
    try:
        with redirect_stdout(io.StringIO()):
            yield
    finally:
        if settrace is not None:
            settrace(tracer)
        sys.breakpointhook, pdb.set_trace = hook, set_trace


def _settrace() -> Callable[[Any], None] | None:
    """
    The way to switch the trace function, or ``None`` when it cannot be paused.

    Under debugpy, ``sys.settrace`` is pydevd's: on 3.11 it still switches tracing
    but warns that the debugger "may stop working", and under ``sys.monitoring``
    (3.12+) it raises — and pausing tracing would not reach the debugger anyway, so
    an IDE breakpoint in ``forward`` also stops once during the export. pydevd's
    own original is used where it keeps one. Looked up in ``sys.modules`` only.
    """
    constants = sys.modules.get("_pydevd_bundle.pydevd_constants")
    if constants is not None and getattr(constants, "PYDEVD_USE_SYS_MONITORING", False):
        return None
    tracing = sys.modules.get("pydevd_tracing")
    holder = getattr(tracing, "TracingFunctionHolder", None)
    return getattr(holder, "_original_tracing", None) or sys.settrace


def _resolve(options: SpecializationOptions | None) -> SpecializationOptions | None:
    """
    The options to load with: the interpreter as ``None``, anything else with debug on.

    The interpreter is refused on the OS framework rather than quietly replaced:
    its interpreter reports no intermediates, and substituting the default options
    would load MPSGraph while every value claimed to come from the interpreter.
    """
    if options is not None:
        return options.with_debug(enabled=True)
    if _uses_os_framework():
        msg = (
            "live() on the interpreter needs the bundled Core AI runtime: the OS "
            "framework's interpreter reports no intermediates. Set USE_LOCAL_COREAI=1 "
            "and start a new process, or pass SpecializationOptions for a device"
        )
        raise ValueError(msg)
    return None


@contextmanager
def live(
    model: torch.nn.Module,
    build: Build | None = None,
    **options: Any,
) -> Generator[LiveSession, None, None]:
    """
    Run *model* eagerly with every tensor carrying its Core AI value as ``.coreai``.

    Inside the block, call the model as usual; set breakpoints anywhere in its
    ``forward``. The first call builds the program from its own arguments; each
    tensor a torch call returns then has a :class:`CoreAIValue` on it, fetched on
    first read. After the block, :meth:`LiveSession.report` still holds the latest
    call's values.

    Args:
        model: The eager module to debug.
        build: ``build(model, args, kwargs) -> (exported_program, program)``, the
            pipeline whose program is being debugged. Defaults to
            :func:`export_and_convert`.
        **options: Passed to :class:`LiveSession` — ``specialization_options``,
            ``function_name``, ``reference``.

    Yields:
        The session.

    """
    session = LiveSession(model, build, **options)
    # First of the model's pre-hooks: the build runs from the first call, and must
    # run before any other hook enters state for that call -- coreai-opt's eager
    # quantization enters its own torch function mode, and expects to leave it.
    hook = model.register_forward_pre_hook(
        session._on_call,  # noqa: SLF001
        with_kwargs=True,
        prepend=True,
    )
    session._hook_reference()  # noqa: SLF001
    try:
        with session._mode:  # noqa: SLF001
            yield session
    finally:
        hook.remove()
        session._unhook_composites()  # noqa: SLF001
        session._unhook_reference()  # noqa: SLF001
