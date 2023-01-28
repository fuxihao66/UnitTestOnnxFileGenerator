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
Example 1 (constant mode): Insert 0 pads to the beginning of the second dimension.

data =
[
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'constant'

constant_value = 0.0

output =
[
    [0.0, 0.0, 1.0, 1.2],
    [0.0, 0.0, 2.3, 3.4],
    [0.0, 0.0, 4.5, 5.7],
]
Example 2 (reflect mode): data = [ [1.0, 1.2], [2.3, 3.4], [4.5, 5.7], ]

pads = [0, 2, 0, 0]

mode = 'reflect'

output =
[
    [1.0, 1.2, 1.0, 1.2],
    [2.3, 3.4, 2.3, 3.4],
    [4.5, 5.7, 4.5, 5.7],
]
Example 3 (edge mode): data = [ [1.0, 1.2], [2.3, 3.4], [4.5, 5.7], ]

pads = [0, 2, 0, 0]

mode = 'edge'

output =
[
    [1.0, 1.0, 1.0, 1.2],
    [2.3, 2.3, 2.3, 3.4],
    [4.5, 4.5, 4.5, 5.7],
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

def create_initializer_scalar(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.AttributeProto.FLOAT
) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=[1],
        vals=tensor_array.flatten().tolist())

    return initializer_tensor

def CreatePadNet(inputDims, outputDims, pad, mode, val, testName, opset) -> None:

    model_input_name = "input1"
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           inputDims)
                                         
    model_output_name = "output1"
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           outputDims)
    
    if opset == 7:
        pad0_node = onnx.helper.make_node(
            name="Pad0",  
            op_type="Pad",
            inputs=[
                model_input_name
            ],
            outputs=[model_output_name],
            pads = pad,
            value = val,
            mode = mode
        )
        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[pad0_node],
            name="PadTest",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                # gather0_indices_initializer_tensor
            ],
        )
    else:
        
        pads_initializer_tensor = create_initializer_tensor(
            name="PadsTensor",
            tensor_array=np.array(pad).astype(np.int64),   # TODO: 64 bit
            data_type=onnx.TensorProto.INT64)
        pads_name = "PadsConst"
        pads_node = onnx.helper.make_node(
            name="Constant0", 
            op_type="Constant",
            inputs=[
            ],
            outputs=[pads_name],
            value = pads_initializer_tensor
        )

        # constant_initializer_tensor = create_initializer_scalar(
        #     name="ValTensor",
        #     tensor_array=np.array(val).astype(np.float32))
        constant_initializer_tensor = create_initializer_tensor(
            name="ValTensor",
            tensor_array=np.array(val).astype(np.float32),
            data_type=onnx.TensorProto.FLOAT)
        constant_name = "ValConst"
        constant_node = onnx.helper.make_node(
            name="Constant1",  # Name is optional.
            op_type="Constant",
            inputs=[
            ],
            outputs=[constant_name],
            # value = np.array([[val]]).astype(np.float32)
            value = constant_initializer_tensor
        )

        pad0_node = onnx.helper.make_node(
            name="Pad0",  # Name is optional.
            op_type="Pad",
            inputs=[
                model_input_name,
                pads_name,
                constant_name,
            ],
            outputs=[model_output_name],
            mode = mode
        )
        graph_def = onnx.helper.make_graph(
            nodes=[pads_node, constant_node, pad0_node],
            name="PadTest",
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
        CreatePadNet([3, 2], [3, 4], [0, 2, 0, 0], 'constant', 0.0, "PadTest0", opset)
        CreatePadNet([3, 2], [3, 4], [0, 2, 0, 0], 'reflect', 0.0, "PadTest1", opset)
        CreatePadNet([3, 2], [3, 4], [0, 2, 0, 0], 'edge', 0.0, "PadTest2", opset)
