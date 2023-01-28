import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnx import load_model
import torch
import numpy as np
import torch.nn as nn
import numpy as np
import onnx

# reference https://leimao.github.io/blog/ONNX-Python-API/


'''
Example 1:

data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]
axes = [0, 1]
starts = [1, 0]
ends = [2, 3]
steps = [1, 2]
result = [
    [5, 7],
]
Example 2:

data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]
starts = [0, 1]
ends = [-1, 1000]
result = [
    [2, 3, 4],
]
'''

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor


def CreateSliceNet(inputDims, outputDims, axes, starts, ends, steps, testName, opset) -> None:

    model_input_name = "input1"
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           inputDims)
                                         
    model_output_name = "output1"
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           outputDims)
    
    if opset == 7:
        if steps != None:
            return
        if axes != None:
            slice0_node = onnx.helper.make_node(
                name="Slice0",  # Name is optional.
                op_type="Slice",
                inputs=[
                    model_input_name
                ],
                outputs=[model_output_name],
                axes = axes,
                starts = starts,
                ends = ends,
            )
        else:
            slice0_node = onnx.helper.make_node(
                name="Slice0",  # Name is optional.
                op_type="Slice",
                inputs=[
                    model_input_name
                ],
                outputs=[model_output_name],
                starts = starts,
                ends = ends,
            )
        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[slice0_node],
            name="SliceTest",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                # gather0_indices_initializer_tensor
            ],
        )
    else:
        if axes != None:
            axes_initializer_tensor = create_initializer_tensor(
            name="AxesTensor",
            tensor_array=np.array(axes).astype(np.int32),
            data_type=onnx.TensorProto.INT32)

            axes_name = "AxesConst"
            axes_node = onnx.helper.make_node(
                name="Constant0",  # Name is optional.
                op_type="Constant",
                inputs=[
                ],
                outputs=[axes_name],
                value = axes_initializer_tensor
            )
        if steps != None:
            steps_initializer_tensor = create_initializer_tensor(
            name="StepsTensor",
            tensor_array=np.array(steps).astype(np.int32),
            data_type=onnx.TensorProto.INT32)

            steps_name = "StepsConst"
            steps_node = onnx.helper.make_node(
                name="Constant1",  # Name is optional.
                op_type="Constant",
                inputs=[
                ],
                outputs=[steps_name],
                value = steps_initializer_tensor
            )
        starts_initializer_tensor = create_initializer_tensor(
            name="StartsTensor",
            tensor_array=np.array(starts).astype(np.int32),
            data_type=onnx.TensorProto.INT32)
        starts_name = "StartsConst"
        starts_node = onnx.helper.make_node(
            name="Constant2",  # Name is optional.
            op_type="Constant",
            inputs=[
            ],
            outputs=[starts_name],
            value = starts_initializer_tensor
        )

        ends_initializer_tensor = create_initializer_tensor(
            name="EndsTensor",
            tensor_array=np.array(ends).astype(np.int32),
            data_type=onnx.TensorProto.INT32)
        ends_name = "EndsConst"
        ends_node = onnx.helper.make_node(
            name="Constant3",  # Name is optional.
            op_type="Constant",
            inputs=[
            ],
            outputs=[ends_name],
            value = ends_initializer_tensor
        )


        if axes != None and steps != None:
            slice0_node = onnx.helper.make_node(
                name="Slice0",  # Name is optional.
                op_type="Slice",
                inputs=[
                    model_input_name,
                    starts_name,
                    ends_name,
                    axes_name,
                    steps_name
                ],
                outputs=[model_output_name],
            )
            graph_def = onnx.helper.make_graph(
                nodes=[starts_node, ends_node, axes_node, steps_node, slice0_node],
                name="SliceTest",
                inputs=[X],  # Graph input
                outputs=[Y],  # Graph output
                initializer=[
                    # gather0_indices_initializer_tensor
                ],
            )
        elif axes != None:
            slice0_node = onnx.helper.make_node(
                name="Slice0",  # Name is optional.
                op_type="Slice",
                inputs=[
                    model_input_name,
                    starts_name,
                    ends_name,
                    axes_name,
                ],
                outputs=[model_output_name],
            )
            graph_def = onnx.helper.make_graph(
                nodes=[starts_node, ends_node, axes_node, slice0_node],
                name="SliceTest",
                inputs=[X],  # Graph input
                outputs=[Y],  # Graph output
                initializer=[
                    # gather0_indices_initializer_tensor
                ],
            )
        else:
            slice0_node = onnx.helper.make_node(
                name="Slice0",  # Name is optional.
                op_type="Slice",
                inputs=[
                    model_input_name,
                    starts_name,
                    ends_name,
                ],
                outputs=[model_output_name],
            )
            graph_def = onnx.helper.make_graph(
                nodes=[starts_node, ends_node, slice0_node],
                name="SliceTest",
                inputs=[X],  # Graph input
                outputs=[Y],  # Graph output
                initializer=[
                    # gather0_indices_initializer_tensor
                ],
            )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = opset

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    onnx.save(model_def, "GeneratedOnnx/FP32/{}-{}.onnx".format(testName, opset))

    fp32_model = load_model("GeneratedOnnx/FP32/{}-{}.onnx".format(testName, opset))
    fp16_model = convert_float_to_float16(fp32_model)
    onnx.save(fp16_model, "GeneratedOnnx/FP16/{}-fp16-{}.onnx".format(testName, opset))

if __name__ == "__main__":
    opsetList = [7, 10, 11, 13]

    for opset in opsetList:
        CreateSliceNet([2, 4], [1, 2], [0, 1], [1, 0], [2, 3], [1, 2], "SliceTest0", opset)
        CreateSliceNet([2, 4], [1, 3], None, [0, 1], [-1, 1000], None, "SliceTest1", opset)
