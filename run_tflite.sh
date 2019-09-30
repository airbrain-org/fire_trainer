#tflite_convert --graph_def_file fire_network.pb --output_file fire_network.tflite --input_arrays=input_1 --output_arrays=final_output_layer
tflite_convert --keras_model_file ./tflite/fire_network.h5 --output_file ./tflite/fire_network.tflite --post_training_quantize true
