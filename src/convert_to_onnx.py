import tensorflow as tf
import tf2onnx

# Load model (FIX)
model = tf.keras.models.load_model(
    "models/lstm_model.keras",
    compile=False
)

print("Input shape:", model.input_shape)

spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13
)

with open("lstm_model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

print("✅ Convert thành công: lstm_model.onnx")