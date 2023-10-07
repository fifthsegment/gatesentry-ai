import numpy as np
from PIL import Image as PILImage
import time
from flask import Flask, request, jsonify
from wand.image import Image as WandImage
import io
import gc
import psutil
import os

interpreter = None
detection_categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
app = Flask(__name__)
loaded_model = False
tflite_model_path = "model2021.tflite"


def load_image_bytes_WAND(file):
    """Load and preprocess the image using Wand."""
    # Convert the image to JPEG format using Wand
    with WandImage(file=file) as img:
        with io.BytesIO() as buffer:
            img.format = 'jpeg'
            img.save(file=buffer)
            buffer.seek(0)
            pil_img = PILImage.open(buffer).convert('RGB')

    # Resize the image to the size expected by the model
    pil_img = pil_img.resize((299, 299))
    
    # Normalize the image and add batch dimension
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def run_inference_2021(file):
    # measure time taken to load the model
    # Load and preprocess image
    with WandImage(file=file) as img:
        with io.BytesIO() as buffer:
            img.format = 'jpeg'
            img.save(file=buffer)
            buffer.seek(0)
            pil_img = PILImage.open(buffer).convert('RGB')

            # Resize the image to the size expected by the model
            pil_img = pil_img.resize((224, 224))
            
            # Normalize the image and add batch dimension
            img_array = np.array(pil_img, dtype=np.float32) / 255.0
            input_data = np.expand_dims(img_array, axis=0)

            # Set input tensor
            input_details = interpreter.get_input_details()
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get output tensor
            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            # print("Duration to run inference = ", time.time() - start)
            return output_data

def run_inference_2019(file):
    # measure time taken to load the model
    # Load and preprocess image
    with WandImage(file=file) as img:
        with io.BytesIO() as buffer:
            img.format = 'jpeg'
            img.save(file=buffer)
            buffer.seek(0)
            pil_img = PILImage.open(buffer).convert('RGB')

            # Resize the image to the size expected by the model
            pil_img = pil_img.resize((299, 299))
            
            # Normalize the image and add batch dimension
            img_array = np.array(pil_img, dtype=np.float32) / 255.0
            input_data = np.expand_dims(img_array, axis=0)

            # Set input tensor
            input_details = interpreter.get_input_details()
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get output tensor
            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            # print("Duration to run inference = ", time.time() - start)
            return output_data
        

def load_model():
    global interpreter
    global loaded_model
    if not loaded_model:
        import tensorflow as tf
        if not tf.__version__.startswith('2'):
            raise ValueError('This code requires TensorFlow V2.x')

        # Set the number of CPUs you want to use
        num_cpus = 2

        # Create a configuration
        config = tf.compat.v1.ConfigProto()
        config.inter_op_parallelism_threads = num_cpus
        config.intra_op_parallelism_threads = num_cpus

        # Set the configuration for the default session
        tf.compat.v1.Session(config=config)
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully")


@app.route('/hello', methods=['GET', 'POST'])
def hello():
    return jsonify({'hello': 'world'})

@app.route('/infer', methods=['POST'])
def api():
    if request.method == 'POST':
        if len(request.files) > 0 :
            load_model()
            file = request.files['image']

            # log time it takes to process the image
            start = time.time()
            # image = Image.open(file)
            # image_array = np.array(image)


            # image_path = "test2.jpg"
            if tflite_model_path == "model2021.tflite":
                output = run_inference_2021(file)
            else:
                output = run_inference_2019(file)
            # Find the index of the maximum probability
            predicted_index = np.argmax(output)

            # Map the index to its corresponding category
            predicted_category = detection_categories[predicted_index]

            # Extract the confidence value for the predicted category
            confidence = output[0][predicted_index]

            memory_used = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # in MB
            response = {
                "category": predicted_category,
                "confidence": round(float(confidence) * 100),  # Ensure it's JSON serializable
                "result": output.tolist(),  # Convert numpy array to list for JSON serialization
                "time": time.time() - start,
                "memory_used": memory_used
            }
             # Force garbage collection
            gc.collect()

            return jsonify(response)
        else:
            return jsonify({'error': 'no file'})
        

# if __name__ == "__main__":
#     tflite_model_path = "model2019.tflite"
#     """Load TFLite model and run inference on the provided image."""
#     # Load TFLite model and allocate tensors
#     interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
#     interpreter.allocate_tensors()
#     app.run(debug=False)


# app.run(debug=False)


# if __name__ == "__main__":
#     tflite_model_path = "model2019.tflite"
#     """Load TFLite model and run inference on the provided image."""
#     # Load TFLite model and allocate tensors
#     # run on port 8000
#     app.run(host='0.0.0.0', port=8000, debug=False)

