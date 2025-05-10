# views.py
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from django.core.files.storage import FileSystemStorage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Ce sera "deepsea_project/outcomes"
#--------------------------------------------------CORAL--------------------------------------------------------------------
model_path = os.path.join(BASE_DIR, 'models', 'coral_reef.keras')
model = load_model(model_path)

def coral_classifier_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']

        # Use Django's FileSystemStorage for saving the uploaded image
        fs = FileSystemStorage(location='media/temp', base_url='/media/temp/')
        filename = fs.save(img.name, img)
        uploaded_image_url = fs.url(filename)

        # Prediction processing
        img_save_path = os.path.join(fs.location, filename)
        img_loaded = image.load_img(img_save_path, target_size=(224, 224))
        img_array = image.img_to_array(img_loaded)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)[0]
        class_names = ['healthy_corals', 'bleached_corals']
        predicted_index = np.argmax(prediction)
        label = 'Healthy' if class_names[predicted_index] == 'healthy_corals' else 'Bleached'
        confidence = round(prediction[predicted_index] * 100, 2)

        return render(request, 'coral_reef.html', {
            'label': label,
            'confidence': confidence,
            'uploaded_image_url': uploaded_image_url
        })

    return render(request, 'coral_reef.html')
#-----------------------------------------------------------------Zooplankton------------------------------------------------------

model_path1 = os.path.join(BASE_DIR, 'models', 'zooplankton.keras')

# Load the model and print confirmation
try:
    model1 = load_model(model_path1)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model loading error: {str(e)}")

# Corrected class names based on training
class_names = [
    'asplanchna', 'bosmina_longirostris', 'calanoid', 'chaoborus', 
    'conochilus', 'cyclops', 'daphnia', 'diaphanosoma', 
    'diaptomus', 'eudiaptomus', 'filinia', 'keratella_cochlearis', 
    'keratella_quadrata', 'kellicottia', 'leptodora_kindtii', 
    'nauplius', 'polyarthra', 'rotifer', 'synchaeta', 'trichocerca', 'alona'
]
def zooplankton_classifier_view(request):
    label = ""
    confidence = 0
    uploaded_image_url = ""

    if request.method == 'POST' and request.FILES.get('image1'):
        try:
            print("Step 1: POST request received with image.")

            # Save the uploaded image
            img = request.FILES['image1']
            fs = FileSystemStorage(location='media/temp', base_url='/media/temp/')
            filename = fs.save(img.name, img)
            uploaded_image_url = fs.url(filename)
            print(f"Step 2: Image saved at {uploaded_image_url}")

            # Image processing
            try:
                img_save_path = os.path.join(fs.location, filename)
                print(f"Step 3: Loading image from {img_save_path}")

                # Ensure the correct image size for DenseNet121
                img_loaded = image.load_img(img_save_path, target_size=(128, 128))
                img_array = image.img_to_array(img_loaded)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                print(f"Step 4: Image preprocessed, shape: {img_array.shape}")

                # Predict the class
                try:
                    prediction = model1.predict(img_array)
                    print(f"Step 5: Prediction shape: {prediction.shape}, Prediction: {prediction}")

                    if prediction is None or len(prediction) == 0 or not np.any(prediction):
                        print("Prediction failed - No output from model.")
                        label = "Prediction Error"
                        confidence = 0
                    else:
                        predicted_index = np.argmax(prediction[0])
                        label = class_names[predicted_index]
                        confidence = round(prediction[0][predicted_index] * 100, 2)
                        print(f"Step 6: Prediction result - Label: {label}, Confidence: {confidence}%")
                except Exception as e:
                    print(f"❌ Prediction error: {str(e)}")
                    label = "Prediction Error"
                    confidence = 0

            except Exception as e:
                print(f"❌ Image processing error: {str(e)}")
                label = "Image Processing Error"
                confidence = 0

        except Exception as e:
            print(f"❌ File upload error: {str(e)}")
            label = "Upload Error"
            confidence = 0

    # Always return the page, even if there is no output or an error
    return render(request, 'zooplankton.html', {
        'label': label,
        'confidence': confidence,
        'uploaded_image_url': uploaded_image_url
    })

# -------------------------------------------------------------------HAB Detection-----------------------------------------------------------------
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from django.core.files.storage import FileSystemStorage

# Load HAB model
hab_model_path = os.path.join(BASE_DIR, 'models', 'model.h5')
hab_model = load_model(hab_model_path)

# Define classes
hab_classes = ["Microalgae", "Harmful Algal Bloom"]

def hab_classifier_view(request):
    label = ""
    confidence = 0
    uploaded_image_url = ""

    if request.method == 'POST' and request.FILES.get('image1'):
        img = request.FILES['image1']
        fs = FileSystemStorage(location='media/temp', base_url='/media/temp/')
        filename = fs.save(img.name, img)
        uploaded_image_url = fs.url(filename)

        img_save_path = os.path.join(fs.location, filename)
        img_loaded = image.load_img(img_save_path, target_size=(180, 180))
        img_array = image.img_to_array(img_loaded)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = hab_model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        label = hab_classes[predicted_index]
        confidence = round(prediction[0][predicted_index] * 100, 2)

    return render(request, 'habdetection.html', {
        'label': label,
        'confidence': confidence,
        'uploaded_image_url': uploaded_image_url
    })
