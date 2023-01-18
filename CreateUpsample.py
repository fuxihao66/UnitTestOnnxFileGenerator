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


def CreateUpsampleNet(inputDims, outputDims, scales, scaleType, testName, opset) -> None:

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
    
    if opset < 9:
        upsample0_node = onnx.helper.make_node(
            name="Upsample0",  # Name is optional.
            op_type="Upsample",
            inputs=[
                model_input_name
            ],
            outputs=[model_output_name],
            scales = scales,
            mode = scaleType
        )
        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[upsample0_node],
            name="UpsampleTest",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                # gather0_indices_initializer_tensor
            ],
        )
    else:
        scales_initializer_tensor = create_initializer_tensor(
            name="ScaleConst",
            tensor_array=scales,
            data_type=onnx.TensorProto.FLOAT)
        constScalesName = "Scales"
        constant0_node = onnx.helper.make_node(
            name="Constant0",  # Name is optional.
            op_type="Constant",
            inputs=[
                
            ],
            outputs=[constScalesName],
            value = scales_initializer_tensor
        )
        upsample0_node = onnx.helper.make_node(
            name="Upsample0",  # Name is optional.
            op_type="Upsample",
            inputs=[
                model_input_name, 
                constScalesName
            ],
            outputs=[model_output_name],
            mode = scaleType
        )
        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[constant0_node, upsample0_node],
            name="UpsampleTest",
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
    opsetList = [7, 9]

    for opset in opsetList:
        CreateUpsampleNet([4], [6], np.array([1.5]), "nearest", "UpsampleTest0", opset)
        CreateUpsampleNet([4], [6], np.array([1.7]), "nearest", "UpsampleTest1", opset)
        CreateUpsampleNet([4], [6], np.array([1.5]), "linear", "UpsampleTest2", opset)
        CreateUpsampleNet([1,3,2,2], [1,3,4,4], np.array([1.,1.,2.,2.]), "linear", "UpsampleTest3", opset)
