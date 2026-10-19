# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Core AI values on the tensors of an eager run: `coreai_torch.debugging.live`."""

from __future__ import annotations

import bdb
import copy
import functools
import inspect
import io
import pdb
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch

import coreai_torch
import coreai_torch.debugging.live as live_module
from coreai_torch import ExternalizeSpec, TorchConverter
from coreai_torch.composite_ops import (
    SDPA,
    GatedDeltaUpdate,
    RMSNorm,
    RMSNormImpl,
    RoPE,
)
from coreai_torch.debugging.live import (
    ATTRIBUTE,
    REFERENCE_ATTRIBUTE,
    Unobservable,
    live,
)

if TYPE_CHECKING:
    from coreai.runtime import SpecializationOptions
    from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def interpreter_only(specialization_options: SpecializationOptions | None) -> None:
    """
    These pin the join — numbering, pieces, composites, debugger safety — and assert
    exact matches, which hold where nothing is fused: the bundled interpreter, the
    suite's default. On a device a delegate fuses, so they would test the delegate.
    """
    if specialization_options is not None:
        pytest.skip("live() is validated on the bundled interpreter")


class Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.proj(x)
        h = torch.relu(pre)
        return h


class Head(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out = torch.nn.Linear(8, 3)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.out(h)
        return y


class TwoSubmodules(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.head = Head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        y = self.head(h)
        return y


class ReadsMidway(torch.nn.Module):
    """Reads a Core AI value inside forward, then calls the same function again."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.seen: list[np.ndarray] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(self.proj(x))
        # Export runs forward too, on fake tensors that carry no value.
        value = getattr(a, ATTRIBUTE, None)
        if value is not None:
            self.seen.append(value.numpy())
        b = torch.relu(a - 0.5)
        return b


def _model() -> TwoSubmodules:
    return TwoSubmodules().eval()


def test_every_tagged_tensor_matches_eager(
    specialization_options: SpecializationOptions | None,
) -> None:
    """Each tensor a torch call returns carries the converted program's value for it."""
    model, x = _model(), torch.randn(2, 4)
    with live(model, specialization_options=specialization_options) as session:
        with torch.no_grad():
            y = model(x)

    report = session.report()
    assert [row["call"] for row in report] == ["linear_1", "relu_1", "linear_2"]
    assert all(row["match"] for row in report), report
    # The result is a graph output, which has no intermediate of its own.
    np.testing.assert_allclose(
        getattr(y, ATTRIBUTE).numpy(), y.numpy(), rtol=1e-4, atol=1e-5
    )


def test_names_are_the_exports_own() -> None:
    """The eager numbering reproduces `torch_fn`, so each call names its last node."""
    model, x = _model(), torch.randn(2, 4)
    with live(model) as session:
        with torch.no_grad():
            model(x)
    nodes = {value.key: value.node for value in session.values}
    exported = {
        node.meta["torch_fn"][0]
        for node in session.exported_program.graph.nodes
        if node.op == "call_function"
    }
    assert set(nodes) <= exported
    # F.linear lowers to a permute and an addmm; the value it returned is the addmm.
    assert nodes["linear_1"].startswith("addmm")


def test_a_line_breakpoint_in_forward_sees_live_tensors() -> None:
    """A debugger stopped on a line of forward finds `coreai` on the tensors in scope."""
    model, x = _model(), torch.randn(2, 4)
    lines, first = inspect.getsourcelines(Encoder.forward)
    line = first + next(i for i, text in enumerate(lines) if "torch.relu(pre)" in text)
    seen: dict[str, np.ndarray] = {}

    class Stop(bdb.Bdb):
        def user_line(self, frame: object) -> None:
            pre = frame.f_locals.get("pre")  # type: ignore[attr-defined]
            if isinstance(pre, torch.Tensor):
                seen["pre"] = getattr(pre, ATTRIBUTE).numpy()
                seen["eager"] = pre.detach().numpy()
            self.set_continue()

    with live(model):
        debugger = Stop()
        debugger.set_break(inspect.getsourcefile(Encoder.forward), line)
        with torch.no_grad():
            debugger.runcall(model, x)

    np.testing.assert_allclose(seen["pre"], seen["eager"], rtol=1e-4, atol=1e-5)


def test_reading_a_value_midway_does_not_shift_the_numbering() -> None:
    """The torch calls a read makes are not counted, so a later relu is still relu_2."""
    model, x = ReadsMidway().eval(), torch.randn(2, 4)
    with live(model) as session:
        with torch.no_grad():
            b = model(x)
    assert len(model.seen) == 1
    assert [value.key for value in session.values] == [
        "linear_1",
        "relu_1",
        "sub_1",
        "relu_2",
    ]
    np.testing.assert_allclose(
        getattr(b, ATTRIBUTE).numpy(), b.numpy(), rtol=1e-4, atol=1e-5
    )
    assert all(row["match"] for row in session.report())


def test_values_follow_the_call_being_debugged() -> None:
    """A second call fetches from its own inputs, not the first call's."""
    model = _model()
    with live(model) as session:
        with torch.no_grad():
            model(torch.randn(2, 4))
            y = model(torch.randn(2, 4))
    np.testing.assert_allclose(
        getattr(y, ATTRIBUTE).numpy(), y.numpy(), rtol=1e-4, atol=1e-5
    )
    assert all(row["match"] for row in session.report())


def test_the_interpreter_is_refused_on_the_os_framework(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading default options instead would claim interpreter values from MPSGraph."""
    monkeypatch.setattr(live_module, "_uses_os_framework", lambda: True)
    with pytest.raises(ValueError, match="USE_LOCAL_COREAI=1"):
        live_module._resolve(None)


def test_a_missing_mapping_is_unobservable_not_wrong() -> None:
    """A call the conversion maps to nothing says so rather than returning a value."""
    model, x = _model(), torch.randn(2, 4)
    with live(model) as session:
        with torch.no_grad():
            model(x)
        key = next(iter(session.origins))
        session.origins[key] = live_module._Origin(session.origins[key].node, None)
        with pytest.raises(Unobservable, match="maps nothing"):
            session.fetch(key, (2, 8))


class AbsLater(torch.nn.Module):
    """Calls `torch.abs` after the line a debugger will stop on."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.proj(x))
        a = torch.abs(h - 1.0)
        return a


def test_what_a_debugger_evaluates_is_not_counted() -> None:
    """
    `p torch.abs(h)` at a pdb prompt must not become the model's `abs_1`.

    Counted, it shifts the model's own `abs` to `abs_2`, which names no node, and
    tags the debugger's result as `abs_1` — reported as a mismatch of 1.0 against
    a value it was never computed from.
    """
    model, x = AbsLater().eval(), torch.randn(2, 4)
    lines, first = inspect.getsourcelines(AbsLater.forward)
    line = first + next(i for i, text in enumerate(lines) if "torch.abs(h" in text)
    commands = io.StringIO("c\np torch.abs(h).sum()\np h.coreai\nc\n")
    with live(model) as session:
        debugger = pdb.Pdb(stdin=commands, stdout=io.StringIO())
        debugger.use_rawinput = False
        debugger.set_break(inspect.getsourcefile(AbsLater.forward), line)
        with torch.no_grad():
            debugger.runcall(model, x)

    assert [value.key for value in session.values] == [
        "linear_1",
        "relu_1",
        "sub_1",
        "abs_1",
    ]
    assert all(row["match"] for row in session.report()), session.report()


class FansOut(torch.nn.Module):
    """Calls whose subgraph has several outputs, a composite, and an elided identity."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.ln = torch.nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b = self.ln(self.lin(x))
        # Unequal pieces: told apart by shape, whatever order they arrive in.
        c, d = torch.split(b, [3, 5], dim=-1)
        values, indices = torch.max(b, dim=-1)
        e = torch.nn.functional.gelu(c).clone()
        return torch.softmax(d, dim=-1), e, values, indices


def test_a_call_is_its_subgraphs_outputs_not_its_last_node() -> None:
    """
    `torch.split` adds one `getitem` per piece; the last node is only the last piece.

    Mapping the call to its last node would hand `c` the value of `d`, and
    `values` the indices.
    """
    model = FansOut().eval()
    exported = torch.export.export(model, (torch.randn(2, 8),))
    groups: dict[str, list[torch.fx.Node]] = {}
    for node in exported.graph.nodes:
        if node.op == "call_function" and "torch_fn" in node.meta:
            groups.setdefault(node.meta["torch_fn"][0], []).append(node)
    split = live_module._results_of(
        groups["split_1"][0].meta["torch_fn"][0], groups["split_1"]
    )
    assert set(split) == {"split_1[0]", "split_1[1]"}, split
    assert split["split_1[1]"] is groups["split_1"][-1]
    assert split["split_1[0]"] is not groups["split_1"][-1]


def test_tuple_results_and_identities_have_their_own_values() -> None:
    """Every piece, both `max` results and a `clone` match."""
    model, x = FansOut().eval(), torch.randn(2, 8)
    with live(model) as session:
        with torch.no_grad():
            model(x)
    report = {row["call"]: row for row in session.report()}
    for call in (
        "split_1[0]",
        "split_1[1]",
        "max_1[0]",
        "max_1[1]",
        "clone_1",
    ):
        assert call in report, sorted(report)
        assert report[call]["match"], report[call]


class EqualPieces(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, d = torch.split(torch.relu(x), 4, dim=-1)
        return torch.softmax(d, dim=-1) + c


def test_pieces_the_runtime_cannot_tell_apart_are_unobservable_not_wrong() -> None:
    """
    The bundled interpreter delivers a multi-output op's outputs in no fixed order.

    Two pieces of one shape and dtype then cannot be told apart, and answering
    by position was wrong in some runs and right in others. Unobservable, never
    a mismatch — repeated, because the failure it guards against is intermittent.
    """
    for _ in range(3):
        model, x = EqualPieces().eval(), torch.randn(2, 8)
        with live(model) as session:
            with torch.no_grad():
                model(x)
        rows = {row["call"]: row for row in session.report()}
        for piece in ("split_1[0]", "split_1[1]"):
            assert rows[piece]["match"] is not False, rows[piece]


def test_a_breakpoint_does_not_stop_during_the_export() -> None:
    """
    `torch.export` runs forward too, on fake tensors with no `coreai`.

    A debugger wrapping the whole session — as an IDE's Debug Cell does — must
    stop on the line once, in the eager run, where the tensor has a value.
    """
    model, x = _model(), torch.randn(2, 4)
    lines, first = inspect.getsourcelines(Encoder.forward)
    line = first + next(i for i, text in enumerate(lines) if "torch.relu(pre)" in text)
    stops: list[bool] = []

    class Stop(bdb.Bdb):
        def user_line(self, frame: object) -> None:
            pre = frame.f_locals.get("pre")  # type: ignore[attr-defined]
            if isinstance(pre, torch.Tensor):
                stops.append(hasattr(pre, ATTRIBUTE))
            self.set_continue()

    def session() -> None:
        with live(model):
            with torch.no_grad():
                model(x)

    debugger = Stop()
    debugger.set_break(inspect.getsourcefile(Encoder.forward), line)
    debugger.runcall(session)
    assert stops == [True]


_COMPOSITE_CALLS_MAPPED = pytest.mark.xfail(
    reason=(
        "a composite's output has no value until Core AI maps the call through the "
        "interpreter's inlining of it (rdar://189232142), which the pinned wheel "
        "predates; remove this marker when the pin moves past it"
    ),
    strict=True,
)


@_COMPOSITE_CALLS_MAPPED
def test_a_composites_output_has_its_value() -> None:
    """The `layer_norm` composite's output matches eager."""
    model, x = FansOut().eval(), torch.randn(2, 8)
    with live(model) as session:
        with torch.no_grad():
            model(x)
    report = {row["call"]: row for row in session.report()}
    assert report["layer_norm_1"]["match"], report["layer_norm_1"]


class SharedNorms(torch.nn.Module):
    """Two `LayerNorm`s of one shape: one composite graph, called twice."""

    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.LayerNorm(8)
        self.second = torch.nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.first(x)
        b = self.second(torch.relu(a))
        return b


def test_a_shared_composite_is_unobservable_not_another_calls_value() -> None:
    """
    The runtime keeps one value per op of a body, from its last call.

    Following the first call into a shared body answered with the second call's
    value — a mismatch of 0.7 on a transformer block. It must not be followed.
    """
    model, x = SharedNorms().eval(), torch.randn(2, 8)
    with live(model) as session:
        with torch.no_grad():
            model(x)
    rows = {row["call"]: row for row in session.report()}
    assert rows["layer_norm_1"]["match"] is not False, rows["layer_norm_1"]
    assert rows["layer_norm_2"]["match"] is not False, rows["layer_norm_2"]


@_COMPOSITE_CALLS_MAPPED
def test_each_call_of_a_shared_composite_has_its_own_value() -> None:
    """Both calls of one shared `layer_norm` body match eager, each its own."""
    model, x = SharedNorms().eval(), torch.randn(2, 8)
    with live(model) as session:
        with torch.no_grad():
            model(x)
    rows = {row["call"]: row for row in session.report()}
    assert rows["layer_norm_1"]["match"], rows["layer_norm_1"]
    assert rows["layer_norm_2"]["match"], rows["layer_norm_2"]


class Linear3D(torch.nn.Module):
    """A `Linear` over a 3-D input, which decomposes to `view → addmm → view`."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return torch.relu(h)


def test_a_value_behind_an_elided_view_is_observable() -> None:
    """The final `view` holds the call's value with no op of its own; its input has it."""
    model, x = Linear3D().eval(), torch.randn(1, 3, 8)
    with live(model) as session:
        with torch.no_grad():
            model(x)
    rows = {row["call"]: row for row in session.report()}
    assert rows["linear_1"]["match"], rows["linear_1"]


class ResidualMLP(torch.nn.Module):
    """An MLP whose last `Linear` feeds the residual add straight away."""

    def __init__(self) -> None:
        super().__init__()
        self.up = torch.nn.Linear(8, 16)
        self.down = torch.nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fed = self.down(torch.nn.functional.gelu(self.up(x)))
        y = x + fed
        return y


def test_the_last_linear_before_a_residual_add_has_its_own_value() -> None:
    """
    The compiler attributes that `addmm` to its own bias add and to the residual add.

    `get_torch_to_coreai_output_mapping` picks the residual add, sees the `add`
    claim it too, and drops the `addmm` — though its bias add holds its value.
    """
    model, x = ResidualMLP().eval(), torch.randn(1, 3, 8)
    with live(model) as session:
        with torch.no_grad():
            model(x)
    rows = {row["call"]: row for row in session.report()}
    assert rows["linear_2"]["match"], rows["linear_2"]
    assert not any(row["match"] is False for row in rows.values()), rows


class Detaches(torch.nn.Module):
    """Calls `detach` twice — the function `live()` itself calls on each input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.detach() * 2.0
        b = (x + 1.0).detach()
        return a + torch.relu(b)


def test_live_s_own_calls_do_not_take_the_model_s_numbers() -> None:
    """
    Starting a call converts its inputs with `detach`, outside any handler.

    Counted, that takes `detach_1`, so the model's first `detach` is named the
    export's second — and tagged with that node's value, a mismatch of 1.0. Two
    things keep it uncounted, and either suffices: the conversion runs paused, and
    the numbering is reset only after it. This fails only when both are lost.
    """
    model, x = Detaches().eval(), torch.randn(2, 3)
    with live(model) as session:
        with torch.no_grad():
            model(x)
    rows = {row["call"]: row for row in session.report()}
    assert not any(row["match"] is False for row in rows.values()), rows
    # `detach_1` aliases the model's input, which has no Core AI op: unobservable.
    # `detach_2` is `(x + 1).detach()`; counted wrongly, it would be the first.
    assert rows["detach_1"]["match"] is None, rows["detach_1"]
    assert rows["detach_2"]["match"], rows["detach_2"]


def test_the_program_is_built_once_from_the_first_call() -> None:
    """The build sees the call being debugged, and later calls reuse its program."""
    model, x = _model(), torch.randn(3, 4)
    seen: list[tuple[torch.Size, ...]] = []

    def build(
        model: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[ExportedProgram, Any]:
        seen.append(tuple(arg.shape for arg in args))
        return live_module.export_and_convert(model, args, kwargs)

    with live(model, build) as session:
        assert session.exported_program is None
        with torch.no_grad():
            model(x)
            model(torch.randn(3, 4))
    assert seen == [(torch.Size([3, 4]),)]
    assert all(row["match"] for row in session.report())


def test_the_builds_own_pipeline_is_the_one_debugged() -> None:
    """
    A build exporting with a dynamic batch serves calls of another batch size.

    The default build would specialize the export to the first call's shape; the
    program has to be the caller's for a second batch size to run at all. The
    bundled interpreter reports no intermediates for a dynamic-shaped program, so
    only the result is observable — the rest must be unobservable, never wrong.
    """
    model = _model()

    def build(
        model: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[ExportedProgram, Any]:
        batch = torch.export.Dim("batch", min=1, max=16)
        exported = torch.export.export(
            model, args, kwargs, dynamic_shapes=({0: batch},)
        ).run_decompositions(coreai_torch.get_decomp_table())
        return exported, TorchConverter().add_exported_program(exported).to_coreai()

    with live(model, build) as session:
        with torch.no_grad():
            model(torch.randn(2, 4))
            y = model(torch.randn(5, 4))
    np.testing.assert_allclose(
        getattr(y, ATTRIBUTE).numpy(), y.numpy(), rtol=1e-4, atol=1e-5
    )
    assert not any(row["match"] is False for row in session.report())


def test_a_program_converted_from_another_export_is_refused() -> None:
    """
    Joined to the wrong export, every name would land on another node's op.

    The undecomposed export has `linear` nodes where the program converted from
    the decomposed one names `addmm` and `permute`.
    """
    model, x = _model(), torch.randn(2, 4)

    def build(
        model: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[ExportedProgram, Any]:
        exported = torch.export.export(model, args, kwargs)
        _, program = live_module.export_and_convert(model, args, kwargs)
        return exported, program

    with live(model, build), pytest.raises(ValueError, match="another export"):
        with torch.no_grad():
            model(x)


def test_a_session_never_called_closes_cleanly() -> None:
    """Nothing is built before the first call, so there is nothing to release."""
    session = live_module.LiveSession(_model())
    assert session.exported_program is None
    session.close()


def test_example_inputs_in_place_of_a_build_are_refused_up_front() -> None:
    """`live(model, (x,))`, the old form, says so rather than failing at the call."""
    with pytest.raises(TypeError, match=r"live\(model\)"):
        live_module.LiveSession(_model(), (torch.randn(2, 4),))  # type: ignore[arg-type]


# -- externalized composites ----------------------------------------------------

_HEADS, _TOKENS, _HEAD_DIM = 2, 4, 8


def _composite_specs() -> list[ExternalizeSpec]:
    return [
        ExternalizeSpec(
            target_class=RMSNormImpl,
            composite_op_name="rms_norm",
            composite_attrs=["axes", "eps"],
        ),
        ExternalizeSpec(
            target_class=RoPE,
            composite_op_name="rope",
            composite_attrs=["scale", "base", "dims", "interleaved"],
        ),
        ExternalizeSpec(
            target_class=SDPA,
            composite_op_name="scaled_dot_product_attention",
            composite_attrs=["scale", "is_causal", "window_size"],
        ),
        ExternalizeSpec(
            target_class=GatedDeltaUpdate,
            composite_op_name="gated_delta_update",
            composite_attrs=["use_qk_l2_norm"],
        ),
    ]


def _externalizing() -> Any:
    return functools.partial(
        live_module.export_and_convert, externalize_modules=_composite_specs()
    )


def _rope_tables() -> tuple[torch.Tensor, torch.Tensor]:
    half = _HEAD_DIM // 2
    freqs = 1.0 / torch.pow(1e4, torch.arange(half) / half)
    theta = torch.arange(_TOKENS).float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cos(theta), torch.sin(theta)


class AttentionBlock(torch.nn.Module):
    """RMSNorm, RoPE on q and k (one module, two calls), SDPA, residual, RMSNorm."""

    def __init__(self) -> None:
        super().__init__()
        width = _HEADS * _HEAD_DIM
        self.norm1 = RMSNorm(width)
        self.qkv = torch.nn.Linear(width, 3 * width)
        self.rope = RoPE()
        self.sdpa = SDPA()
        self.out = torch.nn.Linear(width, width)
        self.norm2 = RMSNorm(width)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        width = _HEADS * _HEAD_DIM
        q, k, v = self.qkv(self.norm1(x)).split(width, dim=-1)
        q, k, v = (
            t.view(1, _TOKENS, _HEADS, _HEAD_DIM).transpose(1, 2) for t in (q, k, v)
        )
        q = self.rope(q, cos=cos, sin=sin)
        k = self.rope(k, cos=cos, sin=sin)
        attended = self.sdpa(q, k, v).transpose(1, 2).reshape(1, _TOKENS, width)
        x = x + self.out(attended)
        return self.norm2(x)


def _attention_session() -> Any:
    model = AttentionBlock().eval()
    cos, sin = _rope_tables()
    with live(model, _externalizing()) as session:
        with torch.no_grad():
            model(torch.randn(1, _TOKENS, _HEADS * _HEAD_DIM), cos, sin)
    return session


def test_externalized_composites_keep_the_numbering() -> None:
    """
    Each composite is one call, as the export records it, however many calls its
    body makes eagerly.

    Counted call by call, RMSNorm's body would advance `mul` and `add` past the
    export's counts, and the residual `add` after it would be named another
    node -- carrying that node's value.
    """
    session = _attention_session()
    rows = {row["call"]: row for row in session.report()}
    assert not any(row["match"] is False for row in rows.values()), rows
    # The calls after every composite are still the export's.
    assert rows["linear_2"]["match"], rows["linear_2"]
    assert rows["add_1"]["match"], rows["add_1"]
    composites = sorted(call for call in rows if ".default_" in call)
    assert len(composites) == 5, composites  # two norms, two rope calls, sdpa


@_COMPOSITE_CALLS_MAPPED
def test_externalized_composite_outputs_have_their_values() -> None:
    """Every composite's output matches eager, both calls of the shared RoPE too."""
    session = _attention_session()
    rows = {row["call"]: row for row in session.report() if ".default_" in row["call"]}
    assert all(row["match"] for row in rows.values()), rows


class TwoResultComposite(torch.nn.Module):
    """GatedDeltaUpdate returns two values; both feed later calls."""

    def __init__(self) -> None:
        super().__init__()
        self.gdu = GatedDeltaUpdate()
        self.out = torch.nn.Linear(_HEAD_DIM, _HEAD_DIM)

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result, state = self.gdu(*inputs)
        return self.out(result), state * 0.5


def test_a_two_result_composite_tags_each_result() -> None:
    """Each result is its own piece of the one call, and the calls after it match."""
    model = TwoResultComposite().eval()
    inputs = (
        *(torch.randn(1, _HEADS, _TOKENS, _HEAD_DIM) for _ in range(3)),
        -torch.rand(1, _HEADS, _TOKENS),
        torch.rand(1, _HEADS, _TOKENS),
        torch.randn(1, _HEADS, _HEAD_DIM, _HEAD_DIM),
    )
    with live(model, _externalizing()) as session:
        with torch.no_grad():
            model(*inputs)
    rows = {row["call"]: row for row in session.report()}
    assert not any(row["match"] is False for row in rows.values()), rows
    pieces = sorted(call for call in rows if ".default_1[" in call)
    assert [piece[-3:] for piece in pieces] == ["[0]", "[1]"], pieces
    assert rows["linear_1"]["match"], rows["linear_1"]


class NormInALoop(torch.nn.Module):
    """One RMSNorm called three times in a loop, then a Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(_HEAD_DIM)
        self.out = torch.nn.Linear(_HEAD_DIM, _HEAD_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.norm(x) * 2.0
        return self.out(x)


def test_a_composite_called_in_a_loop_is_numbered_per_call() -> None:
    """Three calls of one composite are its op's calls 1, 2 and 3."""
    model = NormInALoop().eval()
    with live(model, _externalizing()) as session:
        with torch.no_grad():
            model(torch.randn(2, _HEAD_DIM))
    rows = {row["call"]: row for row in session.report()}
    assert not any(row["match"] is False for row in rows.values()), rows
    norms = sorted(call for call in rows if ".default_" in call)
    assert [norm[-2:] for norm in norms] == ["_1", "_2", "_3"], norms
    assert rows["mul_3"]["match"], rows["mul_3"]
    assert rows["linear_1"]["match"], rows["linear_1"]


# -- state --------------------------------------------------------------------


class CachedLM(torch.nn.Module):
    """The coreai-models signature in miniature: ids, positions, two caches updated in place."""

    def __init__(self, vocab: int = 32, dim: int = 16, max_len: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.out = torch.nn.Linear(dim, vocab)
        self.max_len = max_len

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        h = self.embed(input_ids)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        k_cache.index_copy_(1, position_ids[0], k)
        v_cache.index_copy_(1, position_ids[0], v)
        scores = q @ k_cache.transpose(1, 2) / 4.0
        mask = torch.arange(self.max_len)[None, :] <= position_ids[0][:, None]
        scores = scores.masked_fill(~mask, float("-inf"))
        h = h + torch.softmax(scores, dim=-1) @ v_cache
        return self.out(h)


def test_a_kv_cache_is_state_and_each_step_reads_its_own() -> None:
    """
    The caches are the program's state, and each step starts from what the last left.

    The second step runs on caches the first filled; its values match only if the
    program is given the caches as they were before the step, not as the eager
    forward left them. The `int64` ids are narrowed to the program's `int32`.
    """
    model = CachedLM().eval()
    k_cache, v_cache = torch.zeros(1, 8, 16), torch.zeros(1, 8, 16)
    with live(model) as session, torch.no_grad():
        for start in (0, 3):
            logits = model(
                input_ids=torch.randint(0, 32, (1, 3)),
                position_ids=torch.arange(start, start + 3)[None, :],
                k_cache=k_cache,
                v_cache=v_cache,
            )
            rows = session.report()
            assert not any(row["match"] is False for row in rows), rows
            np.testing.assert_allclose(
                getattr(logits, ATTRIBUTE).numpy(), logits.numpy(), rtol=1e-4, atol=1e-5
            )
    assert int((k_cache.abs().sum(-1) > 0).sum()) == 6


class Accumulator(torch.nn.Module):
    """A state each step adds to: read, modified and written in place."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
        total.add_(self.scale(x))
        return torch.relu(total) * 2.0


def test_every_read_steps_from_the_state_before_the_step() -> None:
    """
    An update that is not idempotent tells a snapshot from the updated state.

    The eager forward adds to `total` in place; a program run on that array would
    add the step twice. Each read must step from a copy taken before forward ran.
    """
    model = Accumulator().eval()
    total = torch.zeros(2, 4)
    with live(model) as session, torch.no_grad():
        for _ in range(3):
            y = model(torch.randn(2, 4), total)
            assert not any(row["match"] is False for row in session.report())
            np.testing.assert_allclose(
                getattr(y, ATTRIBUTE).numpy(), y.numpy(), rtol=1e-4, atol=1e-5
            )


class Rounded(torch.nn.Module):
    """Quantization in miniature: a weight parametrization, as coreai-opt applies one."""

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.round(weight * 4) / 4


def _quantized(model: torch.nn.Module) -> torch.nn.Module:
    """A copy of *model* with every Linear weight rounded, through a parametrization."""
    quantized = copy.deepcopy(model)
    for module in quantized.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.utils.parametrize.register_parametrization(
                module, "weight", Rounded()
            )
    return quantized


def _relative(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((value - reference).norm() / reference.norm())


def test_a_reference_splits_the_error_into_the_models_and_the_lowerings() -> None:
    """
    Against the float original, a module's error is quantization's; against Core AI, conversion's.

    Rounding the weights changes every output, which the model error measures; the
    program computes what the rounded model does, so the lowering error is ~0.
    """
    reference = _model()
    model, x = _quantized(reference), torch.randn(2, 4)
    with live(model, reference=reference) as session, torch.no_grad():
        y = model(x)
    with torch.no_grad():
        expected, quantized = reference.encoder(x), model.encoder(x)

    row = session.compare(model.encoder)
    assert row["module"] == "encoder"
    assert row["model_error"] == pytest.approx(_relative(quantized, expected), rel=1e-5)
    assert row["model_error"] > 1e-3
    assert row["lowering_error"] < 1e-5
    assert row["total_error"] == pytest.approx(row["model_error"], rel=1e-3)
    # Each module's output carries the reference's, and the comparison with it.
    attached = getattr(y, REFERENCE_ATTRIBUTE)
    with torch.no_grad():
        torch.testing.assert_close(attached.tensor, reference(x))
    assert attached.module == ""
    assert attached.compare() == session.compare("")
    assert "lowering error" in repr(attached)
    assert session.compare("encoder") == row
    assert session.compare(reference.encoder) == row


def test_the_reference_takes_no_part_in_the_join() -> None:
    """Its run is uncounted and untagged: every value still matches its own call's."""
    reference = _model()
    model, x = _quantized(reference), torch.randn(2, 4)
    with live(model, reference=reference) as session, torch.no_grad():
        model(x)
    rows = session.report()
    assert any(row["match"] for row in rows), rows
    assert not any(row["match"] is False for row in rows), rows
    theirs = [
        t for calls in session._reference_outputs.values() for c in calls for t in c
    ]  # noqa: SLF001
    assert theirs
    assert not any(hasattr(t, ATTRIBUTE) for t in theirs)


def test_every_module_has_a_row_and_parametrizations_none() -> None:
    reference = _model()
    model = _quantized(reference)
    with live(model, reference=reference) as session, torch.no_grad():
        model(torch.randn(2, 4))
    rows = session.report_modules()
    assert [row["module"] for row in rows] == [
        "",
        "encoder",
        "encoder.proj",
        "head",
        "head.out",
    ]
    assert all(row["model_error"] > 1e-3 for row in rows), rows


def test_weights_are_compared_as_the_model_computes_them() -> None:
    """A parametrized weight is read dequantized; an untouched bias is exact."""
    reference = _model()
    model = _quantized(reference)
    with live(model, reference=reference) as session, torch.no_grad():
        model(torch.randn(2, 4))
    rows = {row["weight"]: row for row in session.report_weights()}
    assert set(rows) == {name for name, _ in reference.named_parameters()}
    weight = reference.encoder.proj.weight.detach()
    assert rows["encoder.proj.weight"]["relative_error"] == pytest.approx(
        _relative(torch.round(weight * 4) / 4, weight), rel=1e-5
    )
    assert rows["encoder.proj.bias"]["relative_error"] == 0.0


class Twice(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.relu(self.proj(x)))


def test_a_module_called_twice_pairs_call_for_call() -> None:
    reference = Twice().eval()
    model, x = _quantized(reference), torch.randn(2, 4)
    with live(model, reference=reference) as session, torch.no_grad():
        model(x)
    with torch.no_grad():
        first, first_ref = model.proj(x), reference.proj(x)
        second = model.proj(torch.relu(first))
        second_ref = reference.proj(torch.relu(first_ref))

    assert session.compare(model.proj, call=0)["model_error"] == pytest.approx(
        _relative(first, first_ref), rel=1e-5
    )
    assert session.compare(model.proj)["model_error"] == pytest.approx(
        _relative(second, second_ref), rel=1e-5
    )
    assert [
        row["call"] for row in session.report_modules() if row["module"] == "proj"
    ] == [0, 1]


def test_the_reference_steps_from_a_copy_of_the_state() -> None:
    """
    The reference runs first, on the same arguments; an in-place update must not reach the model's.

    With identical weights the two agree exactly, on every step, only if each
    starts from the same state -- and the state advances once per step, not twice.
    """
    reference = Accumulator().eval()
    model = copy.deepcopy(reference)
    total, expected = torch.zeros(2, 4), torch.zeros(2, 4)
    with live(model, reference=reference) as session, torch.no_grad():
        for _ in range(3):
            x = torch.randn(2, 4)
            model(x, total)
            expected += reference.scale(x)
            assert session.compare("")["model_error"] == 0.0
    torch.testing.assert_close(total, expected)


def test_a_breakpoint_stops_in_the_models_run_not_the_references() -> None:
    """The two share their classes; only the model's tensors carry `coreai`."""
    reference = _model()
    model, x = _quantized(reference), torch.randn(2, 4)
    lines, first = inspect.getsourcelines(Encoder.forward)
    line = first + next(i for i, text in enumerate(lines) if "torch.relu(pre)" in text)
    stops: list[bool] = []

    class Stop(bdb.Bdb):
        def user_line(self, frame: object) -> None:
            pre = frame.f_locals.get("pre")  # type: ignore[attr-defined]
            if isinstance(pre, torch.Tensor):
                stops.append(hasattr(pre, ATTRIBUTE))
            self.set_continue()

    def session() -> None:
        with live(model, reference=reference), torch.no_grad():
            model(x)

    debugger = Stop()
    debugger.set_break(inspect.getsourcefile(Encoder.forward), line)
    debugger.runcall(session)
    assert stops == [True]


def test_a_reference_with_modules_the_model_lacks_is_refused() -> None:
    with pytest.raises(ValueError, match="paired with the model's by module path"):
        with live(_model(), reference=Head()):
            pass


def test_compare_without_a_reference_says_what_it_needs() -> None:
    model = _model()
    with live(model) as session, torch.no_grad():
        model(torch.randn(2, 4))
    with pytest.raises(ValueError, match="needs a reference model"):
        session.compare(model.encoder)
