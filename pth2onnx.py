import io
import sys
import argparse

import numpy as np
import onnx
import onnxruntime
from onnxsim import simplify
import onnx_graphsurgeon as gs

import torch


class ONNXExporter:
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def run_model(
        self,
        model,
        onnx_path,
        inputs_list,
        dynamic_axes=False,
        tolerate_small_mismatch=False,
        do_constant_folding=True,
        output_names=None,
        input_names=None,
    ):
        model.eval()

        # onnx_io = io.BytesIO()
        onnx_path = onnx_path
        # dynamic = {'images': {0: 'batch'}, 'pred_logits': {0: "batch"}, "pred_boxes": {0: "batch"}}
        # torch.onnx.export(model, inputs_list[0], onnx_io,
        #     input_names=input_names, output_names=output_names,do_constant_folding=True,training=False,opset_version=12)
        torch.onnx.export(
            model,
            inputs_list[0],
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            training=False,
            opset_version=12,
        )

        import onnx
        import onnxsim

        onnx_model = onnx.load(onnx_path)
        onnx_model, _ = onnxsim.simplify(onnx_model)
        onnx.save(onnx_model, onnx_path)

        print(f"[INFO] ONNX model export success! save path: {onnx_path}")

        # validate the exported model with onnx runtime
        for test_inputs in inputs_list:
            with torch.no_grad():
                if isinstance(test_inputs, torch.Tensor) or isinstance(
                    test_inputs, list
                ):
                    # test_inputs = (nested_tensor_from_tensor_list(test_inputs),)
                    test_inputs = (test_inputs,)
                test_ouputs = model(*test_inputs)
                if isinstance(test_ouputs, torch.Tensor):
                    test_ouputs = (test_ouputs,)
            self.ort_validate(
                onnx_path, test_inputs, test_ouputs, tolerate_small_mismatch
            )

        print("[INFO] Validate the exported model with onnx runtime success!")

        # dynamic_shape
        if dynamic_axes:
            # dynamic_axes = [int(ax) for ax in list(dynamic_axes)]
            torch.onnx.export(
                model,
                inputs_list[0],
                "./detr_dynamic.onnx",
                dynamic_axes={
                    input_names[0]: {0: "-1"},
                    output_names[0]: {0: "-1"},
                    output_names[1]: {0: "-1"},
                },
                input_names=input_names,
                output_names=output_names,
                verbose=True,
                opset_version=12,
            )

            print(
                f"[INFO] Dynamic Shape ONNX model export success! Dynamic shape:{dynamic_axes} save path: ./detr_dynamic.onnx"
            )

    def ort_validate(self, onnx_path, inputs, outputs, tolerate_small_mismatch=False):

        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        ort_session = onnxruntime.InferenceSession(onnx_path)
        # compute onnxruntime output prediction
        ort_inputs = dict(
            (ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs)
        )
        ort_outs = ort_session.run(None, ort_inputs)
        for i in range(0, len(outputs)):
            try:
                torch.testing.assert_allclose(
                    outputs[i], ort_outs[i], rtol=1e-02, atol=1e-05
                )
            except AssertionError as error:
                if tolerate_small_mismatch:
                    self.assertIn("(0.00%)", str(error), str(error))
                else:
                    raise

    @staticmethod
    def check_onnx(onnx_path):
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"[INFO]  ONNX model: {onnx_path} check success!")
