# ruff: noqa: UP037

import unittest

import torch_onnx
from torch_onnx import _core
import torch

from onnxscript import FLOAT


bf16 = torch.bfloat16
f64 = torch.float64
f32 = torch.float32
f16 = torch.float16
c32 = torch.complex32
c64 = torch.complex64
c128 = torch.complex128
i8 = torch.int8
i16 = torch.int16
i32 = torch.int32
i64 = torch.int64
b8 = torch.bool
u8 = torch.uint8
u16 = torch.uint16
u32 = torch.uint32
u64 = torch.uint64


class ExportedProgramToIrTest(unittest.TestCase):
    def test_output_metadata_with_tuple_outputs(self):
        class GraphModule(torch.nn.Module):
            def forward(
                self, arg0_1: "f32[4, 3]", arg1_1: "f32[4, 3]", arg2_1: "i64[4]"
            ):
                embedding: "f32[4, 3]" = torch.ops.aten.embedding.default(
                    arg0_1, arg2_1, 1
                )
                arg0_1 = None
                embedding_1: "f32[4, 3]" = torch.ops.aten.embedding.default(
                    arg1_1, arg2_1, 1
                )
                arg1_1 = arg2_1 = None
                return (embedding, embedding_1)

        exported_program = torch.export.export(
            GraphModule(), (torch.rand(4, 3), torch.rand(4, 3), torch.rand(4))
        )
        assert _core.exported_program_to_ir(exported_program)

    def test_inputs_are_wrapped_as_symbolic_tensors_to_support_arithmetic_ops(self):
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: "f32[1]", arg1_1: "f32[1]"):
                add: "f32[1]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
                return add

        def add(a: FLOAT, b: FLOAT, alpha: float = 1) -> FLOAT:
            # Construct model and the ONNX decomposition such that two inputs are added with Python operator
            return a + b

        exported_program = torch.export.export(
            GraphModule(), (torch.rand(1), torch.rand(1))
        )
        print(exported_program)
        registry = torch_onnx.OnnxRegistry.from_torchlib()
        registry.register_op(torch.ops.aten.add.Tensor, add)
        assert _core.exported_program_to_ir(exported_program, registry=registry)


if __name__ == "__main__":
    unittest.main()
