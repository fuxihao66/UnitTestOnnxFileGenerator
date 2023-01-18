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


def CreateAddNet(inputDims, outputDims, testName, opset) -> None:

    # Create a dummy convolutional neural network.

    # IO tensors (ValueInfoProto).
    model_input_name0 = "input0"
    model_input_name1 = "input1"
    X0 = onnx.helper.make_tensor_value_info(model_input_name0,
                                           onnx.TensorProto.FLOAT,
                                           inputDims)
    X1 = onnx.helper.make_tensor_value_info(model_input_name1,
                                           onnx.TensorProto.FLOAT,
                                           inputDims)                                     
    model_output_name = "output1"
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           outputDims)
    
    
    add0_node = onnx.helper.make_node(
        name="Add0",  # Name is optional.
        op_type="Add",
        inputs=[
            model_input_name0,
            model_input_name1
        ],
        outputs=[model_output_name],
    )
    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=[add0_node],
        name="AddTest",
        inputs=[X0, X1],  # Graph input
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
    opsetList = [7, 13, 14]

    for opset in opsetList:
        CreateAddNet([4], [4], "AddTest0", opset)
        CreateAddNet([3, 2], [3, 2], "AddTest1", opset)
        # CreateAddNet([3, 3], [1, 3, 2], np.array([[0, 2]]).astype(np.int32), 1, "GatherTest2", opset)