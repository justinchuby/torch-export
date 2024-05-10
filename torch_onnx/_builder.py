"""Convenience methods for constructing the IR."""

# NOTE: This is a temporary solution for constructing the IR. It should be replaced
# with a more permanent solution in the future.

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

from onnxscript import ir
from onnxscript.ir import _convenience as ir_convenience


class Tape(Iterable[ir.Node]):
    """A tape for recording nodes that are created."""

    def __init__(self) -> None:
        self._nodes: list[ir.Node] = []

    def __iter__(self) -> Sequence[ir.Node]:
        return self._nodes

    @property
    def nodes(self) -> Sequence[ir.Node]:
        return tuple(self._nodes)

    def op(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, ir_convenience.SupportedAttrTypes] | None = None,
        domain: str = "",
    ) -> ir.Value:
        if attributes is None:
            attrs: Sequence[ir.Attr | ir.RefAttr] = ()
        else:
            attrs = ir_convenience.convert_attributes(attributes)
        node = ir.Node(domain, op_type, inputs, attributes=attrs, num_outputs=1)
        self._nodes.append(node)

        return node.outputs[0]

    def op_multi_output(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, ir_convenience.SupportedAttrTypes] | None = None,
        *,
        num_outputs: int,
        domain: str = "",
    ) -> Sequence[ir.Value]:
        if attributes is None:
            attrs: Sequence[ir.Attr | ir.RefAttr] = ()
        else:
            attrs = ir_convenience.convert_attributes(attributes)
        node = ir.Node(
            domain, op_type, inputs, attributes=attrs, num_outputs=num_outputs
        )
        self._nodes.append(node)

        return node.outputs


class OpBuilder:
    def __init__(self, graph: ir.Graph, domain: str, opset_version: int):
        self.graph = graph
        self.domain = domain
        self.opset_version = opset_version
        if (
            imported_version := self.graph.opset_imports.get("domain", opset_version)
        ) != opset_version:
            raise ValueError(
                f"Domain '{domain}' has a different opset version ({imported_version}) imported"
            )
        self.graph.opset_imports[domain] = opset_version
        self._common_constants = {}

    def __getattr__(self, op_type: str) -> Callable:
        return lambda *args, **kwargs: self._make_node(op_type, args, kwargs)

    def _handle_call(
        self, op_type: str, inputs: Sequence[ir.Value], kwargs: dict[str, Any]
    ):
        if op_type == "Constant":
            return self._handle_constant(op_type, inputs, kwargs)
        return self._make_node(op_type, inputs, kwargs)

    def _handle_constant(
        self, op_type: str, inputs: Sequence[ir.Value], kwargs: dict[str, Any]
    ):
        if (value_int := kwargs.pop("value_int", None)) is not None:
            if value_int in self._common_constants:
                return self._common_constants[value_int]
        if (value_ints := kwargs.pop("value_ints", None)) is not None:
            value_ints = tuple(value_ints)
            if value_ints in self._common_constants:
                return self._common_constants[value_ints]
        if (value_float := kwargs.pop("value_float", None)) is not None:
            if value_float in self._common_constants:
                return self._common_constants[value_float]
        if (value_floats := kwargs.pop("value_floats", None)) is not None:
            value_floats = tuple(value_floats)
            if value_floats in self._common_constants:
                return self._common_constants[value_floats]
        return self._make_node(op_type, inputs, kwargs)

    def _make_node(
        self, op_type: str, inputs: Sequence[ir.Value], kwargs: dict[str, Any]
    ):
        self.handle_constant()

        num_outputs = kwargs.pop("num_outputs", 1)
        assert isinstance(num_outputs, int)

        attrs = ir_convenience.convert_attributes(kwargs)
        node = ir.Node(self.domain, op_type, inputs, attributes=attrs, num_outputs=1)
        # Add node to graph
        self.graph.append(node)
        if num_outputs == 1:
            return node.outputs[0]
        return node.outputs
