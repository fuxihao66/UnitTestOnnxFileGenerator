

# traverse every node and find upsample operator
# create a model for every upsample operator
# infer and get upsample scales result
# replace upsample scales with constant and remove redundant operators






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


sess_options = ort.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = "optimized_model.onnx"

# ort_sess = ort.InferenceSession('GeneratedOnnx/FP16/ConvTest0-fp16-7.onnx', providers=['CPUExecutionProvider'])
ort_sess = ort.InferenceSession('D:/UGit/OnnxDMLPlugin/OnnxDMLTest/model/candy-9.onnx', sess_options, providers=['CPUExecutionProvider'])
outputs = ort_sess.run(None, {'input1': x})
output_img = np.transpose(outputs[0].reshape(3, 224, 224), (1, 2, 0)) # float16
output_img *= 255.
output_img = output_img.astype(np.uint8)
output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB )
cv2.imwrite("conv_infer_result_32.png", output_img)