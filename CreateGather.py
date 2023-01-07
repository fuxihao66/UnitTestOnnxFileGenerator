import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
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


def CreateGatherNet(opset) -> None:

    # Create a dummy convolutional neural network.

    # IO tensors (ValueInfoProto).
    model_input_name = "input1"
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           [4])
                                         
    model_output_name = "output1"
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           [4])
    indices = np.array([3,1,0,2]).astype(np.int32)
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
        axis = 0,
    )
    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=[constant0_node, gather0_node],
        name="GatherTest",
        inputs=[X],  # Graph input
        outputs=[Y],  # Graph output
        initializer=[
            gather0_indices_initializer_tensor
        ],
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = opset

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    onnx.save(model_def, "Gather-{}.onnx".format(opset))


if __name__ == "__main__":

    CreateGatherNet(7)
    CreateGatherNet(11)
    CreateGatherNet(13)