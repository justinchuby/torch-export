"""Patch torch.onnx.export to use the exported program"""
import torch

from typing import Union, Any, Optional, Sequence, Mapping, Collection, Type
import io
import torch.export
import logging

logger = logging.Logger(__name__)


def torch_onnx_export_adaptor(
    model: torch.nn.Module,
    args: tuple[Any, ...],
    f: Union[str, io.BytesIO],
    export_params: bool = True,
    verbose: bool = False,
    training: torch.onnx.TrainingMode = torch.onnx.TrainingMode.EVAL,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    operator_export_type: torch.onnx.OperatorExportTypes = torch.onnx.OperatorExportTypes.ONNX,
    opset_version: Optional[int] = None,
    do_constant_folding: bool = True,
    dynamic_axes: Optional[
        Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]
    ] = None,
    keep_initializers_as_inputs: Optional[bool] = None,
    custom_opsets: Optional[Mapping[str, int]] = None,
    export_modules_as_functions: Union[bool, Collection[Type[torch.nn.Module]]] = False,
    autograd_inlining: Optional[bool] = True,
    **_,
) -> None:
    if args and isinstance(args[-1], dict):
        kwargs = args[-1]
        args = args[:-1]
    # TODO: Support dynamic shapes
    program = torch.export.export(model, args, kwargs)
    torch.onnx.dynamo_export(program, *args, **kwargs).save(f)


def patch_torch():
    torch.onnx.export = torch_onnx_export_adaptor
