import onnxruntime as ort
import numpy as np
import cv2

input_img = cv2.imread(r"D:\UGit\OnnxDMLPlugin\OnnxDMLTest\data\testimg.jpg") # h w c
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB )
x = input_img.astype(np.float32)

x /= 255.
# x_linear = np.power((x + 0.055)/1.055, 2.4).astype(np.float32)
# cv2.imwrite("input_hdr.exr", x_linear)
x = np.transpose(x, (2, 0, 1))
x = x.reshape((1, 3, 224, 224))
# ort_sess = ort.InferenceSession('GeneratedOnnx/FP16/ConvTest0-fp16-7.onnx', providers=['CPUExecutionProvider'])
ort_sess = ort.InferenceSession('GeneratedOnnx/FP32/ConvTest0-7.onnx', providers=['CPUExecutionProvider'])
outputs = ort_sess.run(None, {'input1': x})
output_img = np.transpose(outputs[0].reshape(3, 224, 224), (1, 2, 0)) # float16
output_img *= 255.
output_img = output_img.astype(np.uint8)
output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB )
cv2.imwrite("conv_infer_result_32.png", output_img)
