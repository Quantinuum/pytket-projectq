# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This test is adapted primarily from
# https://github.com/ProjectQ-Framework/ProjectQ/blob/develop/examples/variational_quantum_eigensolver.ipynb

import math
import platform
import warnings
from collections import Counter

import numpy as np
import pytest
from hypothesis import given, strategies

from pytket.backends.backend import ResultHandleTypeError
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.backends.resulthandle import ResultHandle
from pytket.backends.status import StatusEnum
from pytket.circuit import BasisOrder, Circuit, OpType, Qubit
from pytket.extensions.projectq import ProjectQBackend
from pytket.passes import BasePass, CliffordSimp, SequencePass
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils.expectations import (
    get_operator_expectation_value,
    get_pauli_expectation_value,
)
from pytket.utils.operators import QubitPauliOperator

warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


# TODO add tests for `get_operator_expectation_value`


def circuit_gen(measure: bool = False) -> Circuit:
    c = Circuit(2)
    c.H(0)
    c.CX(0, 1)
    return c


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_statevector() -> None:
    c = circuit_gen()
    b = ProjectQBackend()
    state = b.run_circuit(c).get_state()
    assert np.allclose(state, [math.sqrt(0.5), 0, 0, math.sqrt(0.5)], atol=1e-10)
    c.add_phase(0.5)
    state1 = b.run_circuit(c).get_state()
    assert np.allclose(state1, state * 1j, atol=1e-10)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.filterwarnings("ignore:Casting complex values")
def test_pauli() -> None:
    c = Circuit(2)
    c.Rz(0.5, 0)
    b = ProjectQBackend()
    zi = QubitPauliString({Qubit(0): Pauli.Z})
    assert np.isclose(get_pauli_expectation_value(c, zi, b), complex(1))
    c.X(0)
    assert np.isclose(get_pauli_expectation_value(c, zi, b), complex(-1))


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.filterwarnings(
    "ignore:Casting complex values to real discards the imaginary part"
)
def test_operator() -> None:
    c = circuit_gen()
    b = ProjectQBackend()
    zz = QubitPauliOperator(
        {QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z]): 1.0}
    )
    assert np.isclose(get_operator_expectation_value(c, zz, b), complex(1.0))
    c.X(0)
    assert np.isclose(get_operator_expectation_value(c, zz, b), complex(-1.0))


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_default_pass() -> None:
    b = ProjectQBackend()
    for ol in range(3):
        comp_pass = b.default_compilation_pass(ol)
        c = Circuit(3, 3)
        c.H(0)
        c.CX(0, 1)
        c.CSWAP(1, 0, 2)
        c.ZZPhase(0.84, 2, 0)
        comp_pass.apply(c)
        for pred in b.required_predicates:
            assert pred.verify(c)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_ilo() -> None:
    b = ProjectQBackend()
    c = Circuit(2)
    c.X(1)
    res = b.run_circuit(c)
    assert np.allclose(
        res.get_state(),
        np.asarray([0, 1, 0, 0], dtype=complex),
        atol=1e-10,
    )
    assert np.allclose(
        res.get_state(basis=BasisOrder.dlo),
        np.asarray([0, 0, 1, 0], dtype=complex),
        atol=1e-10,
    )


def test_swaps_basisorder() -> None:
    # Check that implicit swaps can be corrected irrespective of BasisOrder
    b = ProjectQBackend()
    c = Circuit(4)
    c.X(0)
    c.CX(0, 1)
    c.CX(1, 0)
    CliffordSimp(True).apply(c)
    assert c.n_gates_of_type(OpType.CX) == 1
    c = b.get_compiled_circuit(c)
    res = b.run_circuit(c)
    s_ilo = res.get_state(basis=BasisOrder.ilo)
    s_dlo = res.get_state(basis=BasisOrder.dlo)
    correct_ilo = np.zeros((16,))
    correct_ilo[4] = 1.0
    assert np.allclose(s_ilo, correct_ilo)
    correct_dlo = np.zeros((16,))
    correct_dlo[2] = 1.0
    assert np.allclose(s_dlo, correct_dlo)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_resulthandle() -> None:
    c = Circuit(4, 4).H(0).CX(0, 2)

    b = ProjectQBackend()

    handles = b.process_circuits([c, c.copy()])

    ids = [han[0] for han in handles]

    assert all(isinstance(idval, str) for idval in ids)
    assert ids[0] != ids[1]
    assert len(b.get_result(handles[0]).get_state()) == (1 << 4)
    assert b.circuit_status(handles[1]).status == StatusEnum.COMPLETED
    with pytest.raises(ResultHandleTypeError) as errorinfo:
        _ = b.get_result(ResultHandle("43", 5))
    assert "ResultHandle('43', 5) does not match expected identifier types" in str(
        errorinfo.value
    )

    wronghandle = ResultHandle("asdf")

    with pytest.raises(CircuitNotRunError) as errorinfocirc:
        _ = b.get_result(wronghandle)
    assert (
        f"Circuit corresponding to {wronghandle!r} "  # noqa: ISC003
        + "has not been run by this backend instance."
        in str(errorinfocirc.value)
    )


@pytest.mark.skipif(
    platform.python_version().split(".")[1] == "11",
    reason="Strange AST recursion error.",
)
@given(
    n_shots=strategies.integers(min_value=1, max_value=10),
    n_bits=strategies.integers(min_value=0, max_value=10),
)
def test_shots_bits_edgecases(n_shots, n_bits) -> None:  # type: ignore
    projectq_backend = ProjectQBackend()
    c = Circuit(n_bits, n_bits)

    # TODO TKET-813 add more shot based backends and move to integration tests
    h = projectq_backend.process_circuit(c, n_shots)
    res = projectq_backend.get_result(h)

    correct_shots = np.zeros((n_shots, n_bits), dtype=int)
    correct_shape = (n_shots, n_bits)
    correct_counts = Counter({(0,) * n_bits: n_shots})
    # BackendResult
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts


def test_backend_info() -> None:
    projectq_backend = ProjectQBackend()
    backend_info = projectq_backend.backend_info
    assert backend_info.name == "ProjectQBackend"
    assert backend_info.architecture is None
    assert projectq_backend.characterisation == dict()  # noqa: C408


def test_default_pass_serialization() -> None:
    projectq_backend = ProjectQBackend()

    for opt_level in range(3):
        default_pass = projectq_backend.default_compilation_pass(opt_level)
        original_pass_dict = default_pass.to_dict()
        reconstructed_pass = BasePass.from_dict(original_pass_dict)
        assert isinstance(reconstructed_pass, SequencePass)
        assert original_pass_dict == reconstructed_pass.to_dict()
