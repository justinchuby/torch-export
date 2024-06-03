"""NOTES:

We need a typing module that will handling Python to ONNX type promotion for use.
For example, if we have torch.ops.aten.add(Tensor, 1.0), we need to promote 1.0
to the same type as Tensor. The same thing needs to work for
torch.ops.aten.add(1.0, Tensor) as well, which means we need a mechanism to`
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import onnx
import onnxscript
import torch
from onnxscript import evaluator, ir
from onnxscript.ir import _convenience as ir_convenience

from torch_onnx import _schemas, _tensors

logger = logging.getLogger(__name__)

# TODO(justinchuby): Update ValidAttributeType to ir_convenience.SupportedAttrTypes
ValidAttributeType = (
    ir.TensorProtocol | int | float | bool | str | Sequence[int] | Sequence[float]
)

AllowedArgType = ir.Value | ValidAttributeType


# Logic for adapting inputs from general Python or PyTorch inputs to ONNX ir.Value
def _construct_named_inputs_and_attrs(
    signature: _schemas.OpSignature,
    args: Sequence[AllowedArgType],
    kwargs: Mapping[str, AllowedArgType],
) -> tuple[dict[str, AllowedArgType], dict[str, ValidAttributeType]]:
    """Construct two mappings: name to inputs and named to attributes based on the signature and args/kwargs.

    This function uses the OpSignature to determine which argument in args and kwargs corresponds to
    which parameter in the signature. ONNX node inputs are stored in named_inputs, and attributes are
    stored in named_attrs. If an _optional input_ is not provided, it is filled with None.

    Args:
        signature: The OpSignature for the node.
        args: The positional arguments for the node.
        kwargs: The keyword arguments for the node.

    Returns:
        A tuple of two mappings: named_inputs and named_attrs.

    Raises:
        ValueError: If a required parameter is not provided.
    """
    # 1. Construct the (named_inputs, named_attrs) mapping based on (args, kwargs) and the signature.
    #   a. Loop over all parameters in the signature and args together
    #   b. Depending on param.is_input, Record named_inputs[param.name] = arg or named_attrs[param.name] = arg
    #   c. Handle kwargs as well
    #   d. Fill in None if the input is not provided
    named_inputs = {}
    named_attrs = {}
    reversed_args_stack = list(reversed(args))
    for param in signature.params:
        if isinstance(param, _schemas.Parameter):
            if reversed_args_stack:
                # First exhaust the positional arguments
                named_inputs[param.name] = reversed_args_stack.pop()
            elif param.name in kwargs:
                named_inputs[param.name] = kwargs[param.name]
            elif param.required:
                raise ValueError(
                    f"Required parameter '{param.name}' is not provided. "
                    f"Signature: {signature}. Args: {args}. Kwargs: {kwargs}."
                )
            else:
                logger.debug(
                    "Optional parameter '%s' is not provided. Added as None. Signature: %s",
                    param.name,
                    signature,
                )
                named_inputs[param.name] = None
        else:
            # AttributeParameter
            if reversed_args_stack:
                # First exhaust the positional arguments
                attribute = reversed_args_stack.pop()
            elif param.name in kwargs:
                attribute = kwargs[param.name]
            elif param.default is not None:
                attribute = param.default
            else:
                attribute = None

            if param.required and attribute is None:
                raise ValueError(
                    f"Required attribute '{param.name}' is not provided. "
                    f"Signature: {signature}. Args: {args}. Kwargs: {kwargs}."
                )

            if attribute is None:
                logger.debug(
                    "Optional attribute '%s' is None. Dropped. Signature: %s",
                    param.name,
                    signature,
                )
                continue
            named_attrs[param.name] = attribute
    return named_inputs, named_attrs


def _resolve_parameter_dtypes(
    signature: _schemas.OpSignature, named_inputs: Mapping[str, AllowedArgType]
) -> Mapping[_schemas.TypeConstraintParam, ir.DataType]:
    """Determine which parameter takes which dtype.

    Handle non-tensor input corner cases and type promotion.

    Requires:
        All ir.Value in name_inputs should have dtype set. Their dtype should be
        compatible with the type_constraint of the corresponding parameter in the signature.

    Args:
        signature: The OpSignature for the node.
        named_inputs: The mapping of parameter names to their arguments.

    Returns:
        A mapping of Constraint names to ir.DataType.
    """
    #   a. Create type_binding: dict[str, ir.DataType]
    #   b. Iterate over all named_inputs
    #   b0. Find the corresponding parameter in the signature
    #   b1. If the argument is a Python constant, skip.
    #   b2. If the argument is a ir.Value, Bind {constraint: arg.dtype}.
    type_binding = {}
    for name, arg in named_inputs.items():
        param = signature.params_map[name]
        assert isinstance(
            param, _schemas.Parameter
        ), f"Expected Parameter, got {type(param)}"
        if isinstance(arg, (int, float, bool, str, Sequence, torch.Tensor)):
            # Skip the Python constants because we do not know what dtype they should take yet
            continue
        elif isinstance(arg, ir.Value):
            # NOTE: We assume arg.dtype is compatible with the type_constraint
            assert arg.dtype is not None, f"Expected dtype to be set for {arg}"
            # TODO(justinchuby): Implement type promotion logic here.
            type_binding[param.type_constraint] = arg.dtype
    return type_binding


def _convert_python_constants(
    signature: _schemas.OpSignature,
    named_inputs: dict[str, AllowedArgType],
    type_binding: Mapping[_schemas.TypeConstraintParam, ir.DataType],
    constant_farm: dict[
        tuple[
            bool | int | float | str | ir.TensorProtocol | tuple[int] | tuple[float],
            ir.DataType,
        ],
        ir.Value,
    ],
    opset: onnxscript.values.Opset,
) -> dict[str, ir.Value | None]:
    """Convert Python constants to Constant nodes based on the dtype information.

    The added constants will be replacing values in named_inputs in place.

    Args:
        signature: The OpSignature for the node.
        named_inputs: The mapping of parameter names to their arguments.
        type_binding: A mapping of Constraint names to ir.DataType.
        constant_farm: A dictionary of {(py_value, ir.DataType): ir.Value} to store the deduplicated constants.
        opset: The Opset to use for creating Constant nodes.

    Returns:
        None
    """
    # 3. Convert Python constants to Constant nodes based on the dtype information;
    #    construct sequences
    #   a. Iterate over all parameters in the signature the second time
    #   b. If the parameter is in to_resolve_type:
    #       - If param.constraint in type_binding,
    #         Get the constant from constant_farm (deduplicated);
    #            otherwise set named_inputs[param.name] = Constant(value, dtype=type_binding[param.constraint])
    #       - Otherwise, set named_inputs[param.name] = Constant(value)
    for name, arg in named_inputs.items():
        if isinstance(arg, ir.Value):
            # TODO(justinchuby): Cast the ir.Value here if needed
            continue

        param = signature.params_map[name]
        assert isinstance(
            param, _schemas.Parameter
        ), f"Expected Parameter, got {type(param)}"

        if param.type_constraint in type_binding:
            # A known dtype is available
            dtype = type_binding[param.type_constraint]
        elif len(param.type_constraint.allowed_types) == 1:
            # Only one type is allowed
            dtype = next(iter(param.type_constraint.allowed_types)).dtype
        else:
            # No dtype information available. Infer from the Python constant
            if isinstance(arg, bool):
                dtype = ir.DataType.BOOL
            elif isinstance(arg, float):
                dtype = ir.DataType.FLOAT
            elif isinstance(arg, int):
                dtype = ir.DataType.INT64
            elif isinstance(arg, str):
                dtype = ir.DataType.STRING
            elif isinstance(arg, (tuple, list)) and all(
                isinstance(val, int) for val in arg
            ):
                dtype = ir.DataType.INT64
                # Make the arg hashable
                arg = tuple(arg)
            elif isinstance(arg, (tuple, list)) and all(
                isinstance(val, float) for val in arg
            ):
                dtype = ir.DataType.FLOAT
                # Make the arg hashable
                arg = tuple(arg)
            elif isinstance(arg, (ir.Tensor, ir.TensorProtocol)):
                dtype = arg.dtype
            else:
                raise TypeError(
                    f"Constant input '{arg}' of type '{type(arg)}' is not supported"
                )

        if not isinstance(arg, (ir.Tensor, ir.TensorProtocol)):
            # Deduplicate the constants
            constant_value = constant_farm.get((arg, dtype))
            if constant_value is None:
                constant_tensor = ir.tensor(value=arg, dtype=dtype)
                constant_value = opset.Constant(value=constant_tensor)
                constant_farm[(arg, dtype)] = constant_value
        else:
            constant_value = opset.Constant(value=arg)

        assert (
            constant_value is not None
        ), f"constant_value should not be None here. Arg: {arg}"
        named_inputs[param.name] = constant_value
    return named_inputs  # type: ignore[return-type]


def _construct_node(
    signature: _schemas.OpSignature,
    named_inputs: Mapping[str, ir.Value | None],
    named_attrs: Mapping[str, ValidAttributeType],
    opset: onnxscript.values.Opset,
) -> ir.Node:
    """Construct the node with the inputs and attributes.

    Args:
        signature: The OpSignature for the node.
        named_inputs: The mapping of parameter names to their arguments.
        named_attrs: The mapping of attribute names to their values.
    """
    outputs = [_tensors.SymbolicTensor(opset) for _ in signature.outputs]
    return ir.Node(
        signature.domain,
        signature.name,
        inputs=tuple(named_inputs.values()),
        attributes=ir_convenience.convert_attributes(named_attrs),
        outputs=outputs,
    )


class OpRecorder(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into torchscript."""

    def __init__(
        self, opset: onnxscript.values.Opset, constant_farm: dict[Any, ir.Value]
    ):
        self.nodes = []
        self.opset = opset
        self.functions: dict[ir.OperatorIdentifier, onnxscript.OnnxFunction] = {}
        self.constant_farm = constant_farm

    def _call_op(
        self,
        op_signature: _schemas.OpSignature,
        named_inputs: dict[str, AllowedArgType],
        named_attrs: dict[str, ValidAttributeType],
    ) -> Sequence[_tensors.SymbolicTensor]:
        """Record nodes for the given opschema and arguments.

        Args:
            op_signature: The OpSchema containing the node signature.
            named_inputs: The mapping of parameter names to their arguments.
            named_attrs: The mapping of attribute names to their values.
        """
        type_binding = _resolve_parameter_dtypes(op_signature, named_inputs)
        converted_named_inputs = _convert_python_constants(
            op_signature, named_inputs, type_binding, self.constant_farm, self.opset
        )
        self.nodes.append(
            node := _construct_node(
                op_signature, converted_named_inputs, named_attrs, self.opset
            )
        )
        return node.outputs  # type: ignore

    def eval(
        self,
        schema: onnx.defs.OpSchema,
        args: Sequence[AllowedArgType],
        kwargs: Mapping[str, AllowedArgType],
    ) -> _tensors.SymbolicTensor | Sequence[_tensors.SymbolicTensor]:
        op_signature = _schemas.OpSignature.from_opschema(schema)
        named_inputs, named_attrs = _construct_named_inputs_and_attrs(
            op_signature, args, kwargs
        )
        # TODO(justinchuby): Handle cast
        if schema.name == "CastLike":
            assert len(named_inputs) == 2
            # Skip CastLike if the input and output types are the same
            src_input = named_inputs["input"]
            target_input = named_inputs["target_input"]
            dtypes_available = (
                isinstance(src_input, ir.Value)
                and isinstance(target_input, ir.Value)
                and src_input.dtype is not None
                and target_input.dtype is not None
            )
            if dtypes_available:
                if src_input.dtype == target_input.dtype:
                    # Same type. No cast needed
                    return src_input
                else:
                    # Create a Cast node
                    return self.opset.Cast(src_input, to=target_input.dtype)

        outputs = self._call_op(op_signature, named_inputs, named_attrs)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def eval_function(  # type: ignore[override]
        self,
        function: onnxscript.OnnxFunction,
        args: Sequence[AllowedArgType],
        kwargs: Mapping[str, AllowedArgType],
    ) -> _tensors.SymbolicTensor | Sequence[_tensors.SymbolicTensor] | bool | int:
        # TODO(justinchuby): Pick up from here
        # Special cases for handling IsScalar and Rank
        if function.name == "IsScalar":
            if len(args) != 1:
                raise TypeError(
                    f"Expected 1 positional argument for function '{function}', got {len(args)}."
                )
            if isinstance(args[0], _tensors.SymbolicTensor):
                if args[0].rank is not None:
                    return args[0].rank == 0
                else:
                    # Fall to call add_function_call
                    pass
            elif isinstance(args[0], Sequence):  # noqa: SIM103
                return False
            else:
                # Python constants are scalars
                return True
        if function.name == "Rank":
            if len(args) != 1:
                raise TypeError(
                    f"Expected 1 positional argument for function '{function}', got {len(args)}."
                )
            if isinstance(args[0], _tensors.SymbolicTensor):
                if args[0].rank is not None:
                    return args[0].rank
                else:
                    # Fall to call add_function_call
                    pass
            elif isinstance(args[0], Sequence):
                if all(isinstance(arg, (int, float)) for arg in args[0]):
                    return 1
                else:
                    # Fall to call add_function_call
                    pass
            else:
                # Python constants are scalars
                return 0
        elif function.experimental_traceable:
            # Trace the function call instead of adding the function as a node
            return function.function(*args, **kwargs)

        op_signature = _schemas.OpSignature.from_callable(function.function)
        named_inputs, named_attrs = _construct_named_inputs_and_attrs(
            op_signature, args, kwargs
        )
        outputs = self._call_op(op_signature, named_inputs, named_attrs)

        self.functions[(function.function_ir.domain, function.name, "")] = function
        if len(outputs) == 1:
            return outputs[0]
        return outputs