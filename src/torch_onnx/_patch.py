"""Patch torch.onnx.export to use the exported program"""

from __future__ import annotations

import datetime
import inspect
import io
import logging
import traceback
import warnings
from typing import Any, Mapping, Sequence

import onnx
import torch
import torch.export
from onnxscript import ir

import torch_onnx
from torch_onnx import _analysis, _ir_passes

_BLUE = "\033[96m"
_END = "\033[0m"

logger = logging.getLogger(__name__)


class TorchExportError(RuntimeError):
    """Error during torch.export.export."""

    pass


class OnnxConversionError(RuntimeError):
    """Error during ONNX conversion."""

    pass


def _signature(model) -> inspect.Signature:
    should_be_callable = getattr(model, "forward", model)
    if callable(should_be_callable):
        return inspect.signature(should_be_callable)
    raise ValueError("model has no forward method and is not callable")


def _from_dynamic_axes_to_dynamic_shapes(
    model,
    dynamic_axes=None,
    input_names: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """

    dynamic_axes examples:
    (1) dynamic_axes = {"x": {0: "my_custom_axis_name_1"}, "y": {1: "my_custom_axis_name_2"}}
    (2) dynamic_axes = {"x": [0], "y": [1]}

    these will be converted to dynamic_shapes respectively:
    (1) dynamic_shapes = {"x": {0: Dim("my_custom_axis_name_1")}, "y": {1: Dim("my_custom_axis_name_2")}}
    (2) dynamic_shapes = {"x": {0: Dim("x_dim_0")}, "y": {1: Dim("y_dim_1")}}  # auto-generated dim names

    """
    # https://github.com/pytorch/pytorch/pull/128371
    if dynamic_axes is None:
        return None

    input_names_set = set() if input_names is None else set(input_names)

    dynamic_shapes = {}
    for input_name, axes in dynamic_axes.items():
        if input_name in input_names_set:
            raise ValueError(
                "input names is not supported yet. Please use model forward signature."
            )
        if isinstance(axes, dict):
            dynamic_shapes[input_name] = {
                k: torch.export.Dim(v) for k, v in axes.items()
            }
        elif isinstance(axes, list):
            dynamic_shapes[input_name] = {
                k: torch.export.Dim(f"{input_name}_dim_{k}") for k in axes
            }
        else:
            raise TypeError(
                f"dynamic_axes value must be either a dict or a list, but got {type(axes)}"
            )
    # torch.export.export needs static dim to present in dynamic_shapes
    # for all input tensors, so we need to add them with None
    try:
        sig = _signature(model)
    except ValueError as e:
        warnings.warn(
            f"{e}, skipping auto filling None on static axes...", stacklevel=1
        )
        return dynamic_shapes
    for input_name in sig.parameters:
        if input_name not in dynamic_shapes:
            dynamic_shapes[input_name] = None
    return dynamic_shapes


def torch_onnx_export_adaptor(
    model: torch.nn.Module,
    args: tuple[Any, ...],
    f: str | io.BytesIO,
    *,
    kwargs: dict[str, Any] | None = None,
    export_params: bool = True,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    **_,
) -> ir.Model:
    if not kwargs and args and isinstance(args[-1], dict):
        kwargs = args[-1]
        args = args[:-1]

    dynamic_shapes = _from_dynamic_axes_to_dynamic_shapes(
        model, dynamic_axes, input_names
    )
    try:
        program = torch.export.export(
            model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
        )
    except Exception as e:
        raise TorchExportError(
            "Failed to export the model with torch.export. "
            f"{_BLUE}This is step 1/2{_END} "
            "of exporting the model to ONNX. Please create an issue "
            f"in the PyTorch GitHub repository against the {_BLUE}*torch.export*{_END} component and "
            "attach the full error stack as well as reproduction scripts."
        ) from e

    try:
        ir_model = torch_onnx.exported_program_to_ir(program)

        if input_names:
            _ir_passes.rename_inputs(ir_model, input_names)
        if output_names:
            _ir_passes.rename_outputs(ir_model, output_names)

        if not export_params:
            ir_model.graph.initializers.clear()

        proto = ir.serde.serialize_model(ir_model)
        if proto.ByteSize() >= 1 << 31:
            # TODO: Create an IR pass to handle external tensors conversion
            logger.warning(
                "The serialized ONNX model is larger than 2GB. "
                "Saving the weights in a separate file"
            )
            onnx.save_model(proto, f, save_as_external_data=True)
        else:
            onnx.checker.check_model(proto, full_check=True)
            onnx.save_model(proto, f)

    except Exception as e:
        raise OnnxConversionError(
            "Failed to convert the exported program to an ONNX model. "
            f"{_BLUE}This is step 2/2{_END} "
            "of exporting the model to ONNX. Please create an issue "
            f"in the PyTorch GitHub repository against the {_BLUE}*onnx*{_END} component and "
            "attach the full error stack as well as reproduction scripts. "
            "You can run `torch_onnx.analyze()` to produce an error report after obtaining "
            "an ExportedProgram with `torch.export.export()`."
        ) from e

    return ir_model


def _torch_onnx_export_adapter_with_error_report(
    *args,
    **kwargs,
) -> ir.Model:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    error_report_name = f"{timestamp}.md"
    try:
        torch_onnx_export_adaptor(*args, **kwargs)
    except OnnxConversionError:
        # Run the analysis to get the error report
        model = args[0]
        arg_args = args[1]
        arg_kwargs = kwargs.get("kwargs", {})
        if not arg_kwargs and args and isinstance(arg_args[-1], dict):
            arg_kwargs = arg_args[-1]
            arg_args = arg_args[:-1]
        dynamic_axes = kwargs.get("dynamic_axes")
        if dynamic_axes is not None:
            dynamic_shapes = _from_dynamic_axes_to_dynamic_shapes(model, dynamic_axes)
        else:
            dynamic_shapes = None
        program = torch.export.export(
            model, arg_args, kwargs=arg_kwargs, dynamic_shapes=dynamic_shapes
        )
        with open(f"torch_onnx_error_{error_report_name}", "w") as f:
            f.write("# PyTorch ONNX Conversion Error report\n\n")
            f.write("Error message:\n\n")
            f.write("```\n")
            f.write(traceback.format_exc())
            f.write("```\n\n")
            f.write("Exported program:\n\n")
            f.write("```\n")
            f.write(str(program))
            f.write("```\n\n")
            f.write("## Analysis\n\n")
            _analysis.analyze(program, file=f)
        raise
    except (TorchExportError, Exception):
        with open(f"torch_export_error_{error_report_name}", "w") as f:
            f.write("# PyTorch ONNX Conversion Error report\n\n")
            f.write("torch.export.export error\n\n")
            f.write("Error message:\n\n")
            f.write("```\n")
            f.write(traceback.format_exc())
            f.write("```\n")
        raise


_original_torch_onnx_export = torch.onnx.export
_original_torch_onnx_utils_export = torch.onnx.utils._export


def patch_torch(error_report: bool = False):
    if error_report:
        torch.onnx.export = _torch_onnx_export_adapter_with_error_report
        torch.onnx.utils._export = _torch_onnx_export_adapter_with_error_report
    else:
        torch.onnx.export = torch_onnx_export_adaptor
        torch.onnx.utils._export = torch_onnx_export_adaptor


def unpatch_torch():
    torch.onnx.export = _original_torch_onnx_export
    torch.onnx.utils._export = _original_torch_onnx_utils_export
