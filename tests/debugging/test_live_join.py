# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
The eager-to-export join behind `live()`, checked against the export alone.

`live()` tags each tensor an eager run produces with the export's own name for the
call that made it — `meta["torch_fn"]`, reproduced by counting calls the way the
export's `TorchFunctionMetadataMode` does — and resolves that name to the node
holding the value. Nothing here converts to Core AI: each case runs the eager model
under `live()`'s real tagging mode, runs the exported program under
`TorchFXInspector`, and checks that every tagged tensor equals the value of the node
its key names. So it tests the join, and only the join, on every compute unit.

Each case lists what is expected to stay untagged, and why. A change there — a
PyTorch release renaming `torch_fn`, or a change to `live()`'s rules — fails here
by name rather than quietly tagging less.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import coreai_torch
import coreai_torch.debugging.live as live_module
from coreai_torch.debugging.inspector import TorchFXInspector

# ---------------------------------------------------------------------------
# The cases, one risk each
# ---------------------------------------------------------------------------


class Decomposing(nn.Module):
    """F.linear lowers to permute + addmm; the call's value is the addmm."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 6)

    def forward(self, x):
        h = self.fc(x)
        return torch.relu(h)


class CallOrder(nn.Module):
    """Names follow call order, not use order; a dead call keeps its number."""

    def forward(self, x):
        b = torch.relu(x * 2.0)
        a = torch.relu(x - 1.0)
        unused = torch.relu(x + 5.0)  # noqa: F841
        out = a + b
        for i in range(3):
            out = torch.relu(out - i)
        return out


class ReusedModule(nn.Module):
    """One Linear instance called twice: linear_1 and linear_2."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(torch.tanh(self.fc(x)))


class Nested(nn.Module):
    """Submodules several levels deep, each with its own calls."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(4, 4), nn.GELU()) for _ in range(2)
        )
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


class TupleResults(nn.Module):
    """split (equal and unequal), chunk, unbind, max(dim), topk, sort."""

    def forward(self, x):
        p, q = torch.split(x, 3, dim=-1)
        r, s = torch.split(x, [2, 4], dim=-1)
        u, w = x.chunk(2, dim=0)
        rows = torch.unbind(x, dim=0)
        values, indices = torch.max(x, dim=-1)
        top, where = torch.topk(x, 2, dim=-1)
        ordered, order = torch.sort(x, dim=-1)
        return (
            p + q,
            r.sum() + s.sum(),
            u - w,
            rows[0],
            values,
            indices,
            top,
            where,
            ordered,
            order,
        )


class InPlace(nn.Module):
    """In-place ops return their input; a view of a mutated base is rebuilt."""

    def forward(self, x):
        a = torch.relu(x)
        v = a.view(-1)
        a.add_(1.0)
        a.mul_(2.0)
        b = a.clone()
        b.relu_()
        c = v * 3.0
        return b, c


class SelfReturning(nn.Module):
    """Calls that may return their input unchanged."""

    def forward(self, x):
        a = torch.sigmoid(x)
        b = a.contiguous()
        c = a.to(torch.float32)
        d = a.float()
        e = a.detach()
        return b + c + d + e


class Dunders(nn.Module):
    """Operators, indexing and tensor methods are torch functions too."""

    def forward(self, x, y):
        a = x @ y.T
        b = -a
        c = abs(b) ** 2
        d = c[:, 1:]
        e = d[0]
        f = x.transpose(0, 1).reshape(-1)
        g = f.unsqueeze(0).expand(3, -1)
        return e * 2, g.mean(dim=0), (c > 1.0).float()


class Factories(nn.Module):
    """Tensors made inside forward: zeros_like, arange, ones, full."""

    def forward(self, x):
        z = torch.zeros_like(x)
        r = torch.arange(x.shape[-1], dtype=x.dtype)
        o = torch.ones(x.shape[0], 1)
        f = torch.full_like(x, 0.5)
        return x + z + r + o + f


class ShapeArithmetic(nn.Module):
    """Python arithmetic on shapes, and a branch on a static shape."""

    def forward(self, x):
        batch, width = x.shape
        y = x.reshape(batch * 2, width // 2)
        if y.shape[0] > 1:
            y = torch.cat([y, y], dim=0)
        return y.sum(dim=0)


class Functional(nn.Module):
    """Python-level functions that decompose into many ops."""

    def forward(self, x):
        a = F.layer_norm(x, (x.shape[-1],))
        b = F.softmax(a, dim=-1)
        c = F.gelu(b)
        q = c.unsqueeze(0).unsqueeze(0)
        d = F.scaled_dot_product_attention(q, q, q)
        return F.silu(d.squeeze(0).squeeze(0))


class EvalNorms(nn.Module):
    """BatchNorm and Dropout in eval: batch_norm decomposes, dropout is a no-op."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, 3, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.drop = nn.Dropout(0.5)
        self.emb = nn.Embedding(10, 4)

    def forward(self, image, ids):
        a = self.bn(self.conv(image))
        b = self.drop(a)
        e = self.emb(ids)
        return b.mean(), e


class ManyInOut(nn.Module):
    """Several inputs, a keyword input, and a dict of outputs."""

    def forward(self, x, y, *, scale):
        s = x + y
        t = s * scale
        return {"sum": s, "scaled": torch.tanh(t), "diff": x - y}


class Hooked(nn.Module):
    """A user forward hook that makes torch calls of its own."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.fc.register_forward_hook(
            lambda module, inputs, output: torch.tanh(output) * 0.5
        )

    def forward(self, x):
        return torch.relu(self.fc(x))


class AtenDirect(nn.Module):
    """torch.ops.aten called directly, beside the usual API."""

    def forward(self, x):
        a = torch.ops.aten.relu.default(x)
        b = torch.ops.aten.mul.Tensor(a, a)
        return torch.sigmoid(b)


class Comprehensions(nn.Module):
    """Calls inside a comprehension, a generator and a lambda."""

    def forward(self, x):
        parts = [torch.relu(x + i) for i in range(3)]
        total = sum(p.sum() for p in parts)
        scale = (lambda t: torch.tanh(t))(x)
        return torch.stack(parts).mean(0) * scale + total


class Loops(nn.Module):
    """A ModuleList loop, a nested loop, a fixed while, row iteration, break and continue."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        for i in range(2):
            for j in range(2):
                x = x + torch.tanh(x * (i + j))
        n = 0
        while n < 3:
            x = torch.sigmoid(x) + n
            n += 1
        rows = [torch.exp(row) for row in x]
        for k in range(5):
            if k == 1:
                continue
            if k == 3:
                break
            x = x * torch.cos(x)
        return x + torch.stack(rows)


class StaticIfs(nn.Module):
    """Branches on a Python flag and on shapes, taken and not taken, inside a loop too."""

    def __init__(self, gated=True):
        super().__init__()
        self.gated = gated
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        h = self.fc(x)
        if self.gated:
            h = h * torch.sigmoid(h)
        else:
            h = torch.relu(h)
        if x.shape[0] > 100:
            h = torch.tanh(h)  # never taken for these inputs
        for i in range(3):
            if i % 2 == 0:
                h = h + torch.relu(h - i)
            else:
                h = h - torch.tanh(h)
        return h


class Cond(nn.Module):
    """torch.cond: export traces both branches; eager runs one."""

    def forward(self, x):
        a = torch.relu(x)
        b = torch.cond(
            a.sum() > 0,
            lambda t: torch.sin(t) * 2.0,
            lambda t: torch.cos(t) - 1.0,
            (a,),
        )
        c = torch.relu(b)  # the call after the cond: does its name still line up?
        d = torch.sin(c)  # sin also appears inside a branch
        return d


class DataDependentIf(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return torch.relu(x)
        return torch.tanh(x)


class ShapeBranch(nn.Module):
    """A branch on the input's shape: export specializes to the example's."""

    def forward(self, x):
        if x.shape[0] > 2:
            x = torch.relu(x)
        return torch.tanh(x) + torch.sigmoid(x)


# ---------------------------------------------------------------------------
# The harness: live()'s real tagging mode, against the export only
# ---------------------------------------------------------------------------


class _Recording(live_module._Tagging):
    """`live()`'s mode, also noting tensor-returning calls that got no key."""

    def __torch_function__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        if torch.compiler.is_compiling():
            return func(*args, **(kwargs or {}))
        before = len(self.session.tagged)
        result = super().__torch_function__(func, types, args, kwargs)
        name = getattr(func, "__name__", None)
        if (
            not self.pause_depth
            and name is not None
            and isinstance(result, torch.Tensor)
            and len(self.session.tagged) == before
        ):
            self.session.eager_only.add(f"{name}_{self.counts.get(func, 0)}")
        return result


class _ExportOnly:
    """Stands in for `LiveSession`: the export's keys, and what the eager run tagged."""

    def __init__(self, exported: torch.export.ExportedProgram) -> None:
        groups: dict[str, list[torch.fx.Node]] = {}
        for node in exported.graph.nodes:
            torch_fn = node.meta.get("torch_fn")
            if node.op == "call_function" and torch_fn is not None:
                groups.setdefault(torch_fn[0], []).append(node)
        self.nodes = {
            key: node
            for call, nodes in groups.items()
            for key, node in live_module._results_of(call, nodes).items()
        }
        self.origins = {
            key: live_module._Origin(node.name, None)
            for key, node in self.nodes.items()
        }
        self.mode = _Recording(self)
        self.tagged: list[tuple[str, torch.Tensor]] = []
        self.eager_only: set[str] = set()

    def tag(self, tensor: torch.Tensor, key: str) -> None:
        self.mode.pause_depth += 1  # the clone is ours, not the model's
        try:
            self.tagged.append((key, tensor.detach().clone()))
        finally:
            self.mode.pause_depth -= 1


@dataclass
class Case:
    model: nn.Module
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    eager_only: set[str] = field(default_factory=set)
    """Calls the eager run makes that the export has no node for."""
    export_only: set[str] = field(default_factory=set)
    """Calls the export has a tensor node for that the eager run cannot tag."""
    strict: bool = False
    grad: bool = False
    eager_args: tuple[Any, ...] | None = None


async def _join(
    case: Case,
) -> tuple[int, list[str], set[str], set[str], list[str], Any]:
    case.model.eval()  # not `model = ...`: coreai-opt's graph-mode GraphModule returns None
    model = case.model
    exported = torch.export.export(
        model, case.args, case.kwargs, strict=case.strict
    ).run_decompositions(coreai_torch.get_decomp_table())
    session = _ExportOnly(exported)
    args = case.eager_args if case.eager_args is not None else case.args
    with torch.enable_grad() if case.grad else torch.no_grad(), session.mode:
        model(*args, **case.kwargs)

    names = sorted({node.name for node in session.nodes.values()})
    values = await TorchFXInspector(exported).get_intermediates_for_ops(
        names, (*args, *case.kwargs.values())
    )
    matched, mismatched = 0, []
    for key, eager in session.tagged:
        node = session.nodes[key].name
        got = values.get(node)
        value = None if not got else got[0]
        want = eager.numpy()
        same = (
            value is not None
            and value.shape == want.shape
            and torch.allclose(
                torch.from_numpy(value).double(), eager.double(), rtol=1e-5, atol=1e-6
            )
        )
        if same:
            matched += 1
        else:
            mismatched.append(f"{key} -> {node}")
    # A call counts as covered when either of its keys was tagged: one whose single
    # output is a `getitem` has both `call` and `call[i]`, and only one can match.
    tagged_calls = {key.split("[")[0] for key, _ in session.tagged}
    export_only = {
        key.split("[")[0]
        for key, node in session.nodes.items()
        if isinstance(node.meta.get("val"), torch.Tensor)
    } - tagged_calls
    keys = [key for key, _ in session.tagged]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    return matched, mismatched, session.eager_only, export_only, duplicates


CASES = {
    "decomposing": Case(Decomposing(), (torch.randn(3, 4),)),
    "call order": Case(
        CallOrder(), (torch.randn(2, 3),), eager_only={"add_1", "relu_3"}
    ),
    "call order, new eager inputs": Case(
        CallOrder(),
        (torch.randn(2, 3),),
        eager_only={"add_1", "relu_3"},
        eager_args=(torch.randn(2, 3) * 5,),
    ),
    "reused module": Case(ReusedModule(), (torch.randn(2, 4),)),
    "nested": Case(Nested(), (torch.randn(3, 4),)),
    "nested, strict export": Case(Nested(), (torch.randn(3, 4),), strict=True),
    "nested, grad mode": Case(Nested(), (torch.randn(3, 4),), grad=True),
    "tuple results": Case(
        TupleResults(), (torch.randn(4, 6),), export_only={"unbind_1"}
    ),
    "tuple results, strict export": Case(
        TupleResults(), (torch.randn(4, 6),), export_only={"unbind_1"}, strict=True
    ),
    "in place": Case(InPlace(), (torch.randn(2, 3),), eager_only={"view_1"}),
    "self returning": Case(
        SelfReturning(),
        (torch.randn(2, 3),),
        eager_only={"contiguous_1", "to_1", "float_1"},
    ),
    "dunders": Case(Dunders(), (torch.randn(3, 4), torch.randn(3, 4))),
    "factories": Case(Factories(), (torch.randn(2, 3),)),
    "shape arithmetic": Case(ShapeArithmetic(), (torch.randn(2, 4),)),
    "functional": Case(Functional(), (torch.randn(3, 8),)),
    "eval norms": Case(
        EvalNorms(), (torch.randn(1, 2, 5, 5), torch.tensor([[1, 3, 9]]))
    ),
    "many in, out": Case(
        ManyInOut(),
        (torch.randn(2, 2), torch.randn(2, 2)),
        {"scale": torch.tensor(3.0)},
    ),
    "hooked": Case(Hooked(), (torch.randn(2, 4),)),
    "aten direct": Case(AtenDirect(), (torch.randn(2, 3),)),
    "comprehensions": Case(Comprehensions(), (torch.randn(2, 3),)),
    "loops": Case(Loops(), (torch.randn(3, 4),)),
    "static ifs, flag on": Case(StaticIfs(gated=True), (torch.randn(3, 4),)),
    "static ifs, flag off": Case(StaticIfs(gated=False), (torch.randn(3, 4),)),
    "cond, eager takes the true branch": Case(Cond(), (torch.ones(2, 3),)),
    "cond, eager takes the false branch": Case(
        Cond(), (torch.ones(2, 3),), eager_args=(-torch.ones(2, 3),)
    ),
}
"""
Why each expected gap is expected:

- `call order`: `relu(x + 5)` is dead. Export removes its nodes, but the eager run
  still makes both calls — which is what keeps every later name aligned.
- `in place`: a view of a base mutated later. Functionalization rebuilds the view
  after the mutation, attributed to the call reading it, so the original has no node.
- `self returning`: `contiguous`, `to` and `float` return their input; export makes
  no node, and the input keeps the tag it already had.
- `tuple results`: the decomposition turns the one `unbind` row used into a `select`
  rather than a `getitem`, so the export keys it `unbind_1` while the eager run tags
  tuple pieces `unbind_1[i]`. Untagged, never mistagged.
"""


@pytest.mark.parametrize("name", list(CASES))
async def test_every_tagged_tensor_is_its_nodes_value(name: str) -> None:
    case = CASES[name]
    matched, mismatched, eager_only, export_only, duplicates = await _join(case)
    assert not mismatched, f"tagged with the wrong node: {mismatched}"
    assert matched > 0
    assert not duplicates, f"one key tagged twice in one call: {duplicates}"
    assert eager_only == case.eager_only, (
        f"untagged eager calls changed: {sorted(eager_only)}"
    )
    assert export_only == case.export_only, (
        f"untagged export calls changed: {sorted(export_only)}"
    )


def test_a_data_dependent_branch_is_refused_by_the_export() -> None:
    """Export cannot trace a branch on a value, so there is no path to diverge from."""
    with pytest.raises(Exception, match="data-dependent"):
        torch.export.export(DataDependentIf().eval(), (torch.randn(2, 3),))
