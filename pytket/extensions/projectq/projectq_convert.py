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

from collections.abc import Iterable
from typing import Any

import numpy as np

from projectq import MainEngine  # type: ignore
from projectq import ops as pqo
from projectq.cengines import BasicEngine, LastEngineException  # type: ignore
from projectq.meta import get_control_count  # type: ignore
from projectq.ops._command import Command as ProjectQCommand  # type: ignore
from projectq.ops._command import apply_command
from projectq.types._qubit import Qureg  # type: ignore
from pytket.circuit import Bit, Circuit, Command, Op, OpType
from pytket.passes import AutoRebase
from pytket.transform import Transform

_pq_to_tk_singleqs = {
    pqo.XGate: OpType.X,
    pqo.YGate: OpType.Y,
    pqo.ZGate: OpType.Z,
    pqo.Rx: OpType.Rx,
    pqo.Ry: OpType.Ry,
    pqo.Rz: OpType.Rz,
    pqo.HGate: OpType.H,
    pqo.SGate: OpType.S,
    pqo.TGate: OpType.T,
    pqo.SqrtXGate: OpType.V,
    pqo.MeasureGate: OpType.Measure,
}

# python can't hash projectq controlled gates...
_pq_to_tk_multiqs = {pqo.XGate: OpType.CX, pqo.ZGate: OpType.CZ, pqo.Rz: OpType.CRz}

# Other gates will be added here which are neither controlled operations nor valid tket
# Ops. These gates are currently either ignored (Barrier) or used to determine flushing
# of tketOptimiser (FlushGate)
_OTHER_KNOWN_GATES = {
    pqo.Allocate: OpType.noop,
    pqo.Deallocate: OpType.noop,
    pqo.Barrier: OpType.noop,
    pqo.FlushGate: OpType.noop,
    pqo.SwapGate: OpType.SWAP,
}


_ALLOWED_GATES = {**_pq_to_tk_singleqs, **_pq_to_tk_multiqs, **_OTHER_KNOWN_GATES}
_REBASE = AutoRebase(
    {
        OpType.SWAP,
        OpType.CRz,
        OpType.CX,
        OpType.CZ,
        OpType.H,
        OpType.X,
        OpType.Y,
        OpType.Z,
        OpType.S,
        OpType.T,
        OpType.V,
        OpType.Rx,
        OpType.Ry,
        OpType.Rz,
    },
)
_tk_to_pq_singleqs: dict = {item[1]: item[0] for item in _pq_to_tk_singleqs.items()}
_tk_to_pq_multiqs: dict = {item[1]: item[0] for item in _pq_to_tk_multiqs.items()}


def _get_pq_command_from_tk_command(
    command: Command, engine: MainEngine, container: Any
) -> ProjectQCommand:
    op = command.op
    optype = op.type
    controlled = False
    if optype in _tk_to_pq_singleqs:
        gatetype = _tk_to_pq_singleqs[optype]
    elif optype in _tk_to_pq_multiqs:
        gatetype = _tk_to_pq_multiqs[optype]
        controlled = True
    else:
        raise Exception("Cannot convert op " + str(command) + " to projectq")

    if issubclass(gatetype, pqo.BasicRotationGate):
        params = op.params
        if len(params) != 1:
            raise Exception(f"A Rotation Gate has {len(params)} parameters")
        try:
            gate = gatetype(params[0].evalf() * np.pi)  # type: ignore
        except:  # noqa: E722
            gate = gatetype(params[0] * np.pi)
    elif issubclass(gatetype, pqo.BasicGate):
        gate = gatetype()
    else:
        raise Exception("Gate of type: " + str(gatetype) + " cannot be converted")
    qubs = [q.index[0] for q in command.args]
    if controlled:
        target = container[qubs[-1]]
        qubs.pop()
        controls = (container[i] for i in qubs)
        qubits = gate.make_tuple_of_qureg(target)
        cmd = ProjectQCommand(engine, gate, qubits, controls)
    else:
        qubits = gate.make_tuple_of_qureg(container[i] for i in qubs)
        cmd = ProjectQCommand(engine, gate, qubits)

    return cmd


def tk_to_projectq(
    engine: MainEngine, qureg: Qureg, circuit: Circuit, ignore_measures: bool = False
) -> None:
    """Given a ProjectQ Qureg in an Engine, converts a Circuit to a series of ProjectQ
    Commands on this Qureg.

    :param engine: A ProjectQ MainEngine
    :param qureg: A ProjectQ Qureg in this MainEngine
    :param circuit: A tket Circuit
    """
    if not circuit.is_simple:
        raise Exception("Cannot currently convert non-simple circuits to ProjectQ")
    for command in circuit:
        if ignore_measures and command.op.type == OpType.Measure:
            continue
        if command.op.type == OpType.Barrier:
            continue
        cmd = _get_pq_command_from_tk_command(command, engine, qureg)
        apply_command(cmd)


def _handle_gate(
    command: ProjectQCommand, engine: Any
) -> None:  # must also be a tket Engine
    if command.gate in _OTHER_KNOWN_GATES or type(command.gate) in _OTHER_KNOWN_GATES:
        return
    if (
        type(command.gate) in _pq_to_tk_multiqs
        and len(command.control_qubits) > 0
        and len(command.qubits) > 0
    ):
        engine._translate_multi_qubit_op(command)  # noqa: SLF001
    elif (
        type(command.gate) in _pq_to_tk_singleqs
        and len(command.control_qubits) == 0
        and len(command.qubits) == 1
    ):
        engine._translate_single_qubit_op(command)  # noqa: SLF001
    elif type(command.gate) == pqo.DaggeredGate:  # noqa: E721
        engine._translate_daggered_op(command)  # noqa: SLF001
    else:
        raise Exception(
            "uncaught option "
            + str(command.gate)
            + " controls = "
            + str(len(command.control_qubits))
            + " targets = "
            + str(len(command.qubits))
        )


def _add_daggered_op_to_circuit(cmd: ProjectQCommand, circ: Circuit) -> bool:
    undaggered_gate = cmd.gate.get_inverse()
    if type(undaggered_gate) == pqo.TGate:  # noqa: E721
        op = Op.create(OpType.Tdg)
    elif type(undaggered_gate) == pqo.SGate:  # noqa: E721
        op = Op.create(OpType.Sdg)
    else:
        raise Exception("cannot recognise daggered op of type " + str(cmd.gate))
    qubit_no = cmd.qubits[0][0].id
    assert len(cmd.qubits) == 1
    assert len(cmd.qubits[0]) == 1
    new_qubit = False
    if qubit_no >= circ.n_qubits:
        circ.add_blank_wires(1 + qubit_no - circ.n_qubits)
        new_qubit = True
    circ.add_gate(Op=op, args=[qubit_no])
    return new_qubit


def _add_single_qubit_op_to_circuit(cmd: ProjectQCommand, circ: Circuit) -> bool:
    assert len(cmd.qubits) == 1
    assert len(cmd.qubits[0]) == 1
    qubit_no = cmd.qubits[0][0].id
    new_qubit = False
    if get_control_count(cmd) > 0:
        raise Exception(
            "singleq gate "
            + str(cmd.gate)
            + " has "
            + str(get_control_count(cmd))
            + " control qubits"
        )
    if qubit_no >= circ.n_qubits:
        circ.add_blank_wires(1 + qubit_no - circ.n_qubits)
        new_qubit = True
    if type(cmd.gate) == pqo.MeasureGate:  # noqa: E721
        bit = Bit("c", qubit_no)
        if bit not in circ.bits:
            circ.add_bit(bit)
        circ.Measure(qubit_no, qubit_no)
        return new_qubit
    if type(cmd.gate) in (pqo.Rx, pqo.Ry, pqo.Rz):
        op = Op.create(_pq_to_tk_singleqs[type(cmd.gate)], cmd.gate.angle / np.pi)
    else:
        op = Op.create(_pq_to_tk_singleqs[type(cmd.gate)])
    circ.add_gate(Op=op, args=[qubit_no])
    return new_qubit


def _add_multi_qubit_op_to_circuit(cmd: ProjectQCommand, circ: Circuit) -> list:
    assert len(cmd.qubits) > 0
    qubs = [qb for qr in cmd.all_qubits for qb in qr]
    if get_control_count(cmd) < 1:
        raise Exception("multiq gate " + str(cmd.gate) + " has no controls")
    new_qubits = []
    for q in qubs:
        qubit_no = q.id
        if qubit_no >= circ.n_qubits:
            circ.add_blank_wires(1 + qubit_no - circ.n_qubits)
            new_qubits.append(q)
    if type(cmd.gate) == pqo.CRz:  # noqa: E721
        op = Op.create(_pq_to_tk_multiqs[type(cmd.gate)], cmd.gate.angle / np.pi)
    else:
        op = Op.create(_pq_to_tk_multiqs[type(cmd.gate)])
    qubit_nos = [qb.id for qr in cmd.all_qubits for qb in qr]
    circ.add_gate(Op=op, args=qubit_nos)
    return new_qubits


class tketBackendEngine(BasicEngine):
    """
    A projectq backend designed to translate from projectq commands
    to tket Circuits
    """

    def __init__(self) -> None:
        """
        Initialize the tketBackendEngine.

        Initializes local Circuit to an empty Circuit.
        """
        BasicEngine.__init__(self)
        self._circuit = Circuit()

    @property
    def circuit(self) -> Circuit:
        """
        Returns the tket Circuit accumulated so far by the engine.

        :raises NotImplementedError: If the Circuit has no gates, assumes user forgot to
            flush engines.

        :return: The tket Circuit from the engine.
        """
        if self._circuit.n_gates == 0:
            raise NotImplementedError(
                "Circuit has no gates. Have you flushed your engine?"
            )
        return self._circuit

    def is_available(self, cmd: ProjectQCommand) -> bool:
        """
        Ask the next engine whether a command is available, i.e.,
        whether it can be executed by the next engine(s).

        :raises projectq.cengines.LastEngineException: If is_last_engine is True but is_available is not
            implemented.

        :param cmd: Command for which to check availability.

        :return: True if the command can be executed.
        """
        try:
            return bool(BasicEngine.is_available(self, cmd))
        except LastEngineException:
            return True

    def receive(self, command_list: Iterable) -> None:
        """Process commands from a list and append to local Circuit."""
        for cmd in command_list:
            _handle_gate(cmd, self)

    def _translate_daggered_op(self, cmd: ProjectQCommand) -> None:
        # assume it is a single qubit op
        _add_daggered_op_to_circuit(cmd, self._circuit)

    def _translate_single_qubit_op(self, cmd: ProjectQCommand) -> None:
        _add_single_qubit_op_to_circuit(cmd, self._circuit)

    def _translate_multi_qubit_op(self, cmd: ProjectQCommand) -> None:
        _add_multi_qubit_op_to_circuit(cmd, self._circuit)


class tketOptimiser(BasicEngine):
    """
    A ProjectQ BasicEngine designed to translate from ProjectQ commands
    to tket Circuits, optimise them, and then return other ProjectQ commands.
    """

    def __init__(self) -> None:
        BasicEngine.__init__(self)
        self._circuit = Circuit()
        self._qubit_dictionary: dict = dict()  # noqa: C408

    def receive(self, command_list: list) -> None:
        """
        Receives a list of commands and appends to local Circuit. If a flush gate is
        received, optimises the Circuit using a default Transform pass and then sends
        the commands from this optimised Circuit into the next engine.
        """
        for cmd in command_list:
            if cmd.gate == pqo.FlushGate():  # flush gate --> optimize and then flush
                cmd_list = self._optimise()
                cmd_list.append(cmd)
                self._circuit = Circuit()
                self._qubit_dictionary = dict()  # noqa: C408
                self.send(cmd_list)
                continue

            _handle_gate(cmd, self)

    def _optimise(
        self,
    ) -> list:
        # takes the circuit and optimises it before regurgitating it as a series of
        # ProjectQ commands
        if self._circuit.n_qubits != 0:
            Transform.OptimisePhaseGadgets().apply(self._circuit)
            _REBASE.apply(self._circuit)

        cmd_list = []

        for i in range(self._circuit.n_qubits):
            gate = pqo.Allocate
            cmd = ProjectQCommand(
                self.main_engine,
                gate,
                gate.make_tuple_of_qureg(self._qubit_dictionary[i]),
            )
            cmd_list.append(cmd)

        if self._circuit.n_gates == 0:
            return cmd_list
        for command in self._circuit:
            cmd = _get_pq_command_from_tk_command(
                command, self.main_engine, self._qubit_dictionary
            )
            cmd_list.append(cmd)
        return cmd_list

    def _translate_daggered_op(self, cmd: ProjectQCommand) -> None:
        # assume it is a single qubit op, as the only daggered ops which are of the
        # ProjectQ DaggeredGate class are single qubit
        new_qubit = _add_daggered_op_to_circuit(cmd, self._circuit)
        # if this qubit hasn't been seen before by the circuit, add to dictionary
        if new_qubit:
            self._qubit_dictionary[cmd.qubits[0][0].id] = cmd.qubits[0][0]

    def _translate_single_qubit_op(self, cmd: ProjectQCommand) -> None:
        new_qubit = _add_single_qubit_op_to_circuit(cmd, self._circuit)
        if new_qubit:
            self._qubit_dictionary[cmd.qubits[0][0].id] = cmd.qubits[0][0]

    def _translate_multi_qubit_op(self, cmd: ProjectQCommand) -> None:
        new_qubits = _add_multi_qubit_op_to_circuit(cmd, self._circuit)
        for q in new_qubits:
            self._qubit_dictionary[q.id] = q
