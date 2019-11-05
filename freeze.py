import numpy as np
import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def save_to_tflite(model, directory_name, file_name, dataset, do_quantize):
    # TODO-JYW: tflite from TF 2.0 is not working with the keras model.
    # Continue working with the TF 1.15.rc2.  

    if (do_quantize):
        image_batch, _ = next(dataset)
        def representative_dataset_gen():
            for input_value in image_batch:
                # Convert the shape of input_value to 1,160,160,3 from 160,160,3.               
                yield [input_value.reshape(1,input_value.shape[0],input_value.shape[1],
                    input_value.shape[2])]

        converter = tf.lite.TFLiteConverter.from_keras_model_file(directory_name + file_name)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()
        full_file_name = directory_name + "quant_" + file_name + ".tflite"
        open(full_file_name, "wb").write(tflite_model)
        print("tflite model file saved to: ", full_file_name)

def save_to_pb(session, model, file_name):
    frozen_graph = freeze_session(session, output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, './', file_name + '.pb', as_text=False)    

def test_freeze():
    X = np.array([[0,0], [0,1], [1,0], [1,1]], 'float32')
    Y = np.array([[0], [1], [1], [0]], 'float32')

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, input_dim=2, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

    model.fit(X, Y, batch_size=1, nb_epoch=100, verbose=0)

    # inputs:  ['dense_input']
    print('inputs: ', [input.op.name for input in model.inputs])

    # outputs:  ['dense_4/Sigmoid']
    print('outputs: ', [output.op.name for output in model.outputs])

    model.save('./xor.h5')

    frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, './', 'xor.pbtxt', as_text=True)
    tf.train.write_graph(frozen_graph, './', 'xor.pb', as_text=False)