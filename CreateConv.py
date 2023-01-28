import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnx import load_model
import torch
import numpy as np
import torch.nn as nn
import numpy as np
import onnx


width = 224
height = 224
# width = 720
# height = 480
mdevice = "cuda:0"
x=torch.randn(1,3,height,width).to(mdevice)


class ConvTestNet(nn.Module):
    # a tinyer Unet which only has 3 downsample pass
    def __init__(self):
        super(ConvTestNet, self).__init__()

        self.conv0 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
       
    def forward(self, inputx):
        
        return self.conv0(inputx)

model = ConvTestNet()
model = model.to(mdevice)
model.eval()
input_names = [ "input1"]
output_names = [ "output1"]

opset_list = [7, 11]

for opset in opset_list:
    torch.onnx.export(model, (x),
                    "GeneratedOnnx/FP32/ConvTest0-{}.onnx".format(opset), verbose=True, input_names=input_names, output_names=output_names,opset_version=opset)
    fp32_model = load_model("GeneratedOnnx/FP32/ConvTest0-{}.onnx".format(opset))
    fp16_model = convert_float_to_float16(fp32_model)
    onnx.save(fp16_model, "GeneratedOnnx/FP16/ConvTest0-fp16-{}.onnx".format(opset))