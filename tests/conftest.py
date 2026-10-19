# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Root test configuration."""

import os

import pytest

_COMPUTE_UNIT_KIND_CHOICES = ("interpreter", "cpu", "gpu", "neural_engine")
_COMPUTE_UNIT_KIND_DEFAULT = "interpreter"


def _compute_unit_kind_from_argv(argv: list[str]) -> str:
    """The ``--compute-unit-kind`` on the command line, before pytest has parsed it."""
    for index, arg in enumerate(argv):
        if arg.startswith("--compute-unit-kind="):
            return arg.split("=", 1)[1]
        if arg == "--compute-unit-kind" and index + 1 < len(argv):
            return argv[index + 1]
    return _COMPUTE_UNIT_KIND_DEFAULT


def _pin_runtime(compute_unit_kind: str) -> None:
    """Select the Core AI runtime for *compute_unit_kind*; see `pytest_configure`."""
    if compute_unit_kind == "interpreter":
        os.environ.setdefault("USE_LOCAL_COREAI", "1")
    else:
        os.environ.pop("USE_LOCAL_COREAI", None)


# Pinned at import as well as in `pytest_configure`, because configure is too late
# whenever a path is given: pytest imports the conftests along it first, and
# `tests/debugging/conftest.py` imports `tests/utils.py`, which imports
# `coreai.runtime` — the moment the runtime is chosen. `pytest tests/debugging/...`
# therefore ran its "interpreter" tests on the OS framework, where default options
# load MPSGraph. This conftest is imported before any below it.
_pin_runtime(_compute_unit_kind_from_argv(__import__("sys").argv))

_current_test_id: str = ""


@pytest.fixture(autouse=True)
def update_current_test_id(request):
    """Automatically updates the unique test ID before each test runs."""
    global _current_test_id
    _current_test_id = request.node.nodeid


def get_current_test_id() -> str:
    return _current_test_id


_dump_optests = False


def dump_optests_enabled() -> bool:
    return _dump_optests


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options."""
    parser.addoption(
        "--compute-unit-kind",
        choices=list(_COMPUTE_UNIT_KIND_CHOICES),
        default=_COMPUTE_UNIT_KIND_DEFAULT,
        help=(
            "Compute unit used by validate_numerical_output:\n"
            "  interpreter (default) - bundled runtime (USE_LOCAL_COREAI=1)\n"
            "  cpu                   - SpecializationOptions.cpu_only() (BNNS)\n"
            "  gpu                   - preferred ComputeUnitKind.gpu() (MPSGraph)\n"
            "  neural_engine         - preferred ComputeUnitKind.neural_engine()\n"
            "Anything other than 'interpreter' unsets USE_LOCAL_COREAI so the OS\n"
            "runtime is used."
        ),
    )
    parser.addoption(
        "--dump-optests",
        action="store_true",
        default=False,
        help="Trigger optest dumping",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Publish the selected compute unit to the test utils.

    For ``--compute-unit-kind=interpreter`` we pin ``USE_LOCAL_COREAI=1`` so the
    bundled runtime is used. For any real compute unit (cpu/gpu/neural_engine)
    we drop the env var so the OS runtime — which actually exposes those
    compute units — gets picked up.
    """
    compute_unit_kind = config.getoption("--compute-unit-kind")
    _pin_runtime(compute_unit_kind)

    from tests import utils

    utils.set_test_compute_unit_kind(compute_unit_kind)
    global _dump_optests
    _dump_optests = config.getoption("--dump-optests")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip control-flow tests on non-interpreter compute units.

    Higher-order ops like ``torch.cond`` / ``while_loop`` are not yet supported
    by the cpu/gpu/neural_engine compute unit runtimes. Mark such tests with
    ``@pytest.mark.control_flow`` and they'll be auto-skipped whenever
    ``--compute-unit-kind`` is not ``interpreter``.
    """
    compute_unit_kind = config.getoption("--compute-unit-kind")
    if compute_unit_kind == "interpreter":
        return
    skip_marker = pytest.mark.skip(
        reason=f"control_flow ops not supported on --compute-unit-kind={compute_unit_kind}"
    )
    for item in items:
        if "control_flow" in item.keywords:
            item.add_marker(skip_marker)
