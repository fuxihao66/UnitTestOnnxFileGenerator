import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnx import load_model
import torch
import numpy as np
import torch.nn as nn
import numpy as np
import onnx

'''

Test 1:
data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  indices = [
      [0, 1],
      [1, 2],
  ]
  output = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]
Test 2:
data = [
      [1.0, 1.2, 1.9],
      [2.3, 3.4, 3.9],
      [4.5, 5.7, 5.9],
  ]
  indices = [
      [0, 2],
  ]
  axis = 1,
  output = [
        [[1.0, 1.9]],
        [[2.3, 3.9]],
        [[4.5, 5.9]],
    ]
'''

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


def CreateGatherNet(inputDims, outputDims, indices, gatherAxis, testName, opset) -> None:

    # Create a dummy convolutional neural network.

    # IO tensors (ValueInfoProto).
    model_input_name = "input1"
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           inputDims)
                                         
    model_output_name = "output1"
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           outputDims)
    # indices = np.array([i % 4 for i in range(6)]).astype(np.int32)
    gather0_indices_initializer_tensor = create_initializer_tensor(
        name="GatherIndices0",
        tensor_array=indices,
        data_type=onnx.TensorProto.INT32)

    constIndicesName = "Indices"
    constant0_node = onnx.helper.make_node(
        name="Constant0",  # Name is optional.
        op_type="Constant",
        inputs=[
            
        ],
        outputs=[constIndicesName],
        value = gather0_indices_initializer_tensor
    )

    
    gather0_node = onnx.helper.make_node(
        name="Gather0",  # Name is optional.
        op_type="Gather",
        inputs=[
            model_input_name, constIndicesName
        ],
        outputs=[model_output_name],
        axis = gatherAxis,
    )
    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=[constant0_node, gather0_node],
        name="GatherTest",
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
    opsetList = [7, 11, 13]

    for opset in opsetList:
        CreateGatherNet([4], [6], np.array([i % 4 for i in range(6)]).astype(np.int32), 0, "GatherTest0", opset)
        CreateGatherNet([3, 2], [2, 2, 2], np.array([[0, 1],[1, 2]]).astype(np.int32), 0, "GatherTest1", opset)
        CreateGatherNet([3, 3], [3, 1, 2], np.array([[0, 2]]).astype(np.int32), 1, "GatherTest2", opset)