from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the segmentation model
segmentation_model_edv = load_model('G:\MyDocuments\Desktop\App\segmentation_model_edv.h5')
segmentation_model_esv = load_model('G:\MyDocuments\Desktop\App\segmentation_model_esv.h5')

# Load ESV and EDV models
edv_model = load_model('G:\MyDocuments\Desktop\App\densenet_edv_prediction_model.h5')
esv_model = load_model('G:\MyDocuments\Desktop\App\densenet_esv_prediction_model.h5')
def prediction(input_image):
  input_image=cv2.imread(input_image)
  input_image1 = input_image.reshape(1, 112, 112, 3)
  predict=model.predict(input_image1)
  
  threshold = 0.5
  binary_mask = (predict > threshold).astype(np.uint8)
  segmented_image = (predict * 255).astype(np.uint8)
  cv2.imwrite("segment.jpg",segmented_image[0])

def extract_green(image_path, output_path):
    
    original_image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])  
    upper_green = np.array([80, 255, 255]) 
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    green_part = cv2.bitwise_and(original_image, original_image, mask=green_mask)
    cv2.imwrite(output_path, green_part)

def overlay_green_mask(background_image_path, mask_image_path, output_path):

    background_image = cv2.imread(background_image_path)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
    mask_image = cv2.resize(mask_image, (background_image.shape[1], background_image.shape[0]))
    inverted_mask = cv2.bitwise_not(mask_image[:, :, 1])
    background_without_green = cv2.bitwise_and(background_image, background_image, mask=inverted_mask)
    result = cv2.add(background_without_green, mask_image)
    cv2.imwrite(output_path, result)

def preprocess_image(image):
    # Add preprocessing logic for your segmentation model
    # (resize, normalize, etc.)
    # ...
    f=image.reshape(1, 112, 112, 3)
    return f

def calculate_ef(edv, esv):
    ef = ((edv - esv) / edv) * 100
    return ef
def status(ef):
    if(ef>=52 and ef<=72):
        return "Normal"
    elif ef>=41 and ef<=53:
        return "Mildly Abnormal"
    elif ef>=30 and ef<=40:
        return "Moderately Abnoraml"
    else:
      return "Severely Abnormal"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get video file from the request
        video_file = request.files['video']

        # Save the video file to a temporary location
        video_path = 'temp_video.mp4'
        video_file.save(video_path)

        # Read the video file
        video = cv2.VideoCapture(video_path)

        # Extract the first 3 frames
        frames = [video.read()[1] for _ in range(3)]
        
        # Perform segmentation on each frame
        segmented_frames_edv = []
        segmented_frames_esv = []
        for frame in frames:
            preprocessed_frame = preprocess_image(frame)
            segmentation_result_edv = segmentation_model_edv.predict(preprocessed_frame)
            segmentation_result_esv = segmentation_model_esv.predict(preprocessed_frame)
            threshold = 0.5
            binary_mask = (segmentation_result_edv > threshold).astype(np.uint8)
            segmented_image = (segmentation_result_edv* 255).astype(np.uint8)
            cv2.imwrite("segment.jpg",segmented_image[0])
            
            extract_green("segment.jpg","mask.jpg")
            cv2.imwrite("frame.jpg",frames[0])
            overlay_green_mask("frame.jpg","mask.jpg", "annotated_edv.jpg")
            an_edv=cv2.imread("annotated_edv.jpg")

            threshold = 0.5
            binary_mask = (segmentation_result_esv > threshold).astype(np.uint8)
            segmented_image = (segmentation_result_esv* 255).astype(np.uint8)
            cv2.imwrite("segment.jpg",segmented_image[0])
            
            extract_green("segment.jpg","mask.jpg")
            cv2.imwrite("frame.jpg",frames[0])
            overlay_green_mask("frame.jpg","mask.jpg", "annotated_esv.jpg")
            
            an_esv=cv2.imread("annotated_esv.jpg")
            segmented_frames_edv.append(an_edv)
            segmented_frames_esv.append(an_esv)

        # Use ESV and EDV models to predict values
        esv_predictions = esv_model.predict(np.array(segmented_frames_esv))
        edv_predictions = edv_model.predict(np.array(segmented_frames_edv))

        # Calculate EF values
        ef_value = calculate_ef(max(edv_predictions),min(esv_predictions))
        ef_status=status(ef_value)
        # Determine if values are normal or abnormal (add your criteria)
        result = {'EF': ef_value[0], 'EDV': max(edv_predictions)[0], 'ESV': min(esv_predictions)[0],'EF_Status':ef_status}

        # Remove the temporary video file
        video.release()
        
        return render_template('index.html', result=result)

    return render_template('index.html')

    

if __name__ == '__main__':
    app.run(debug=True)

