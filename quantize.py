import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model("Models/accent")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()