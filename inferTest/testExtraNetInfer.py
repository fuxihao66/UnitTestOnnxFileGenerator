

import onnxruntime as ort
import numpy as np
import cv2
import time


# x /= 255.
# x_linear = np.power((x + 0.055)/1.055, 2.4).astype(np.float32)
# cv2.imwrite("input_hdr.exr", x_linear)
# x = np.zeros((1, 21, 720, 1280), dtype=np.float16)
x = np.zeros((1, 21, 720, 1280), dtype=np.float32)

sess_options = ort.SessionOptions()
# sess_options.enable_profiling = True
# Set graph optimization level
# sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "D:/optimized_model_opset9_fp32.onnx"

ort_sess = ort.InferenceSession('D:/UGit/ExtraNetTRTInference-main/UNetGated_Opset9.onnx', sess_options, providers=["DmlExecutionProvider" ])#CPUExecutionProvider
# ort_sess = ort.InferenceSession('D:/UGit/OnnxDMLPlugin/OnnxDMLTest/model/candy-9.onnx', providers=["CPUExecutionProvider" ])#
# ort_sess = ort.InferenceSession('D:/UGit/OnnxDMLPlugin/OnnxDMLTest/model/candy-9.onnx', providers=["CUDAExecutionProvider" ])#
start = time.time()
for i in range(1000):
    outputs = ort_sess.run(None, {'input_buffer': x})

# prof_file = ort_sess.end_profiling()
# print(prof_file)
print("used time = {}".format((time.time() - start)))


# output_img = np.transpose(outputs[0].reshape(3, 224, 224), (1, 2, 0)) 
# # output_img *= 255.
# output_img = output_img.astype(np.uint8)
# output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB )
# cv2.imwrite("conv_infer_result_32.png", output_img)