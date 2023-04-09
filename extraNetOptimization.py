
import onnxruntime as ort
import numpy as np
import cv2
x = np.zeros((1, 21, 720, 1280), dtype=np.float16)

sess_options = ort.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = "D:/optimized_model.onnx"

# ort_sess = ort.InferenceSession('GeneratedOnnx/FP16/ConvTest0-fp16-7.onnx', providers=['CPUExecutionProvider'])
ort_sess = ort.InferenceSession('D:/UNetGated_UEIntegrate_fp16.onnx', sess_options, providers=['CPUExecutionProvider'])
outputs = ort_sess.run(None, {'input_buffer': x})
# output_img = np.transpose(outputs[0].reshape(3, 224, 224), (1, 2, 0)) # float16
# output_img *= 255.
# output_img = output_img.astype(np.uint8)
# output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB )
# cv2.imwrite("conv_infer_result_32.png", output_img)