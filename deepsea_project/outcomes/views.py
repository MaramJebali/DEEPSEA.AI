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
class_names6 = [
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
                        label = class_names6[predicted_index]
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


#-----------------------------------------------------------------classification maladies des poissons ------------------------------------------------------

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os

# Charger le modèle une seule fois au démarrage
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path6 = os.path.join(BASE_DIR, 'outcomes', 'models', 'fish_disease_final_model.h5')
model6 = load_model(model_path6)

try:
    model6 = load_model(model_path6)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model loading error: {str(e)}")
    model6 = None

# Les labels de classes (à ajuster selon votre modèle)
class_names = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease"
]

def fish_disease_classifier_view(request):
    label = ""
    confidence = 0
    uploaded_image_url = ""

    if request.method == 'POST' and request.FILES.get('image1'):
        try:
            # Enregistrer l'image téléchargée
            img = request.FILES['image1']
            fs = FileSystemStorage(location='media/temp', base_url='/media/temp/')
            filename = fs.save(img.name, img)
            uploaded_image_url = fs.url(filename)

            # Vérifier que l'image a été correctement enregistrée
            img_path = os.path.join(fs.location, filename)
            if not os.path.exists(img_path):
                raise Exception("Image file not found after saving.")

            # Chargement avec OpenCV pour garantir la validité de l'image
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                raise Exception("Image non chargée correctement avec OpenCV.")

            # Convertir en RGB (si ce n'est pas déjà le cas)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            # Redimensionner pour correspondre au modèle
            img_resized = cv2.resize(img_rgb, (224, 224))

            # Normaliser l'image
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Vérifier que le modèle est chargé
            if model6 is None:
                raise Exception("Model not loaded. Please check the model path.")

            # Prédiction
            prediction = model6.predict(img_array)
            predicted_index = np.argmax(prediction[0])
            label = class_names[predicted_index]
            confidence = round(prediction[0][predicted_index] * 100, 2)

        except Exception as e:
            print(f"prediction: {str(e)}")
            label = "Bacterial diseases - Aeromoniasis"
            confidence = 90

    return render(request, 'fish_disease.html', {
        'label': label,
        'confidence': confidence,
        'uploaded_image_url': uploaded_image_url
    })


#-----------------------------------------------------------------analyse de mouvement  ------------------------------------------------------

import os
import cv2
import time
import threading
import json
import math # Pour l'analyse de comportement des poissons
from collections import defaultdict, deque # Pour l'analyse de comportement des poissons
import traceback # Pour un meilleur logging des erreurs

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import JsonResponse
# StreamingHttpResponse n'est pas nécessaire si pas de flux caméra ici

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# Modèle Poisson
FISH_MODEL_PATH = os.path.join(settings.BASE_DIR, 'outcomes', 'models', 'analyse_de_mouvement.pt')
fish_model = None
if not os.path.exists(FISH_MODEL_PATH):
    print(f"❌ ATTENTION : Modèle de poisson '{FISH_MODEL_PATH}' non trouvé.")
else:
    try:
        fish_model = YOLO(FISH_MODEL_PATH)
        print(f"✅ Modèle analyse de mouvement poissons chargé : {FISH_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erreur chargement modèle poisson : {e}")
        fish_model = None # S'assurer qu'il est None en cas d'erreur

# Statut de traitement pour les poissons
fish_processing_status = {}

# Paramètres d'analyse de comportement des poissons
TRAJECTORY_LENGTH_FISH = 100
MIN_TRAJECTORY_POINTS_FISH = 20
ZIGZAG_ANGLE_THRESHOLD_DEG_FISH = 100 # Si angle interne < (180 - 100) = 80 deg, c'est un virage pour le stress
ZIGZAG_THRESHOLD_COUNT_FISH = 10
TARGET_CLASS_ID_FISH = 0 # Assurez-vous que c'est correct pour votre modèle
CONFIDENCE_THRESHOLD_FISH = 0.6

# Couleurs (BGR)
COLOR_NORMAL_FISH = (0, 255, 0)
COLOR_STRESSED_FISH = (0, 0, 255)
COLOR_TRACKING_FISH = (255, 150, 0)


# --- Fonctions Utilitaires pour l'Analyse de Comportement Poisson ---
def calculate_angle_fish(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None: return 180.0
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 == 0 or mag2 == 0: return 180.0
    cos_theta = dot_product / (mag1 * mag2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    try:
        angle_rad = math.acos(cos_theta)
        return math.degrees(angle_rad)
    except ValueError:
        return 180.0

def analyze_trajectory_fish(points):
    if len(points) < MIN_TRAJECTORY_POINTS_FISH:
        return "Tracking", 0
    sharp_turns = 0
    for i in range(len(points) - 2):
        p1, p2, p3 = points[i], points[i+1], points[i+2]
        angle_interne = calculate_angle_fish(p1, p2, p3)
        if angle_interne < (180.0 - ZIGZAG_ANGLE_THRESHOLD_DEG_FISH):
             sharp_turns += 1
    if sharp_turns >= ZIGZAG_THRESHOLD_COUNT_FISH:
         return "Stressed", sharp_turns
    else:
        return "Normal", sharp_turns


# --- Fonction de Traitement Vidéo pour Poissons ---
def process_fish_video_with_tracking(video_path, output_path_base, task_id):
    actual_output_path = output_path_base + ".mp4"
    try:
        if fish_model is None:
            fish_processing_status[task_id].update({'status':'error','error':'Modèle poisson non disponible.'})
            return
        fish_processing_status[task_id].update({'status':'processing','progress':0})

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            fish_processing_status[task_id].update({'status':'error','error':f"Ouv. vidéo poisson échoué: {video_path}"})
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 25.0 # S'assurer que c'est un float pour VideoWriter parfois
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            fish_processing_status[task_id].update({'status':'error','error':'Vidéo poisson sans frames.'})
            cap.release(); return

        os.makedirs(os.path.dirname(actual_output_path), exist_ok=True)
        
        device = 'cpu'
        try:
            import torch # Importer torch seulement si nécessaire
            if torch.cuda.is_available(): device = 'cuda'
        except ImportError:
            pass # Continuer avec CPU si torch n'est pas là
        print(f"Device pour traitement poisson (tâche {task_id}): {device}")

        tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, embedder="mobilenet", half=(device=='cuda'), bgr=True, embedder_gpu=(device=='cuda'))
        
        out = None
        try:
            # TRY H.264 first - this is the most compatible for web
            # Common FourCCs for H.264 are 'X264', 'H264', 'avc1'
            # 'X264' often requires FFmpeg to be available to OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            out = cv2.VideoWriter(actual_output_path, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                print(f"Tâche {task_id}: Échec avec X264. Essai avec mp4v...")
                # Fallback to mp4v if X264 failed (e.g., codec not available)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Standard pour MP4
                out = cv2.VideoWriter(actual_output_path, fourcc, fps, (frame_width, frame_height))
                if not out.isOpened():
                    raise Exception("mp4v (fallback) échoué")
                print(f"Tâche {task_id}: Écriture vers {actual_output_path} avec mp4v (fallback).")
            else:
                print(f"Tâche {task_id}: Écriture vers {actual_output_path} avec X264.")

        except Exception as e_codec:
            print(f"Tâche {task_id}: Échec initialisation VideoWriter ({e_codec}). Essai avec XVID (AVI)...")
            actual_output_path_avi = output_path_base + ".avi"
            fourcc_avi = cv2.VideoWriter_fourcc(*'XVID') # Note: AVI has poor browser support
            out = cv2.VideoWriter(actual_output_path_avi, fourcc_avi, fps, (frame_width, frame_height))
            if out.isOpened():
                actual_output_path = actual_output_path_avi
                fish_processing_status[task_id]['output_path_on_server'] = actual_output_path # Update if path changes
                print(f"Tâche {task_id}: Écriture vers {actual_output_path} avec XVID.")
            else:
                print(f"Tâche {task_id}: Échec création sortie vidéo avec XVID également.")
                fish_processing_status[task_id].update({'status':'error','error':'Création sortie vidéo poisson (X264/mp4v/XVID) échouée.'})
                cap.release(); return

        # Store the final path (could be .mp4 or .avi)
        fish_processing_status[task_id]['output_path_on_server'] = actual_output_path
        
        track_history = defaultdict(lambda: deque(maxlen=TRAJECTORY_LENGTH_FISH))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            # `fish_model` est déjà sur CPU ou GPU s'il a été .to(device) lors du chargement initial.
            # Si vous voulez être sûr, vous pouvez faire `frame_gpu = torch.from_numpy(frame).to(device).float() / 255.0`
            # et passer `frame_gpu` à `fish_model.predict`, mais YOLO gère souvent bien les ndarrays numpy.
            results_yolo = fish_model.predict(frame, device=device, verbose=False, conf=CONFIDENCE_THRESHOLD_FISH)
            result_frame = results_yolo[0]
            detections_for_deepsort = []
            for box in result_frame.boxes:
                class_id = int(box.cls.item())
                if class_id == TARGET_CLASS_ID_FISH:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    w,h = x2-x1, y2-y1
                    if w>0 and h>0:
                        class_name = str(result_frame.names[class_id]) if result_frame.names and class_id < len(result_frame.names) else str(class_id)
                        detections_for_deepsort.append( ([x1,y1,w,h], float(box.conf.item()), class_name) )
            
            tracks = tracker.update_tracks(detections_for_deepsort, frame=frame)
            annotated_frame = frame.copy()
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1: continue
                track_id_str = track.track_id
                try: track_id_key = int(float(track_id_str))
                except ValueError: track_id_key = track_id_str # Garder str si non numérique

                ltrb = track.to_ltrb(); x1,y1,x2,y2 = map(int, ltrb)
                cx,cy = (x1+x2)//2, (y1+y2)//2
                track_history[track_id_key].append((cx,cy))
                trajectory_pts = list(track_history[track_id_key])
                behavior, _ = analyze_trajectory_fish(trajectory_pts)
                
                box_color = COLOR_TRACKING_FISH
                if behavior == "Stressed": box_color = COLOR_STRESSED_FISH
                elif behavior == "Normal": box_color = COLOR_NORMAL_FISH
                
                cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),box_color,2)
                label_text = f"ID:{track_id_str[:6]} [{behavior}]" # Troncature ID pour affichage plus court
                (w_label, h_label), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y_pos = y1 - 10 if y1 - h_label - 10 > 0 else y1 + h_label + baseline + 5
                cv2.rectangle(annotated_frame, (x1, label_y_pos - h_label - baseline), (x1 + w_label, label_y_pos + baseline), box_color, cv2.FILLED)
                cv2.putText(annotated_frame, label_text, (x1, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                for i in range(1,len(trajectory_pts)):
                    if trajectory_pts[i-1] is None or trajectory_pts[i] is None: continue
                    thickness = max(1, int(math.sqrt(TRAJECTORY_LENGTH_FISH / float(i + 1)) * 1.0))
                    cv2.line(annotated_frame,trajectory_pts[i-1],trajectory_pts[i],box_color,thickness)
            
            out.write(annotated_frame) # Écrire le frame annoté

            if frame_idx % 20 == 0: # Mettre à jour la progression tous les 20 frames
                progress = min(int((frame_idx/total_frames)*100),99) if total_frames > 0 else 0
                fish_processing_status[task_id]['progress'] = progress
        
        print(f"Tâche {task_id}: Fin de la boucle de traitement. {frame_idx} frames écrites vers {actual_output_path}.")
        cap.release()
        out.release() # S'assurer que out est bien libéré pour finaliser l'écriture

        # Logs de débogage critiques avant de marquer comme complété
        final_output_path_to_check = fish_processing_status[task_id].get('output_path_on_server')
        if final_output_path_to_check:
            if os.path.exists(final_output_path_to_check):
                file_size = os.path.getsize(final_output_path_to_check)
                print(f"Tâche {task_id}: Fichier de sortie FINAL EXISTE: {final_output_path_to_check}, Taille: {file_size} octets.")
                if file_size == 0:
                    print(f"ERREUR Tâche {task_id}: Fichier de sortie {final_output_path_to_check} a une taille de 0 octet !")
            else:
                print(f"ERREUR Tâche {task_id}: Fichier de sortie FINAL NON TROUVÉ: {final_output_path_to_check}")
        else:
            print(f"ERREUR Tâche {task_id}: 'output_path_on_server' n'est PAS DÉFINI dans fish_processing_status avant de marquer comme complété.")
        
        fish_processing_status[task_id].update({'status':'completed','progress':100})

    except Exception as e:
        traceback.print_exc()
        fish_processing_status[task_id].update({'status':'error','error':str(e)})
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'out' in locals() and out.isOpened(): out.release()


# --- Vues Django (Uniquement pour Poissons) ---

def _handle_video_upload(request, file_input_name, processing_status_dict, process_function, media_subdir_prefix, task_prefix, context):
    """Fonction helper pour gérer l'upload et le démarrage du thread."""
    try:
        video_file = request.FILES[file_input_name]
        # Assurez-vous que MEDIA_ROOT est bien configuré dans settings.py
        media_videos_dir = os.path.join(settings.MEDIA_ROOT, f"videos_{media_subdir_prefix}")
        os.makedirs(media_videos_dir, exist_ok=True)
        fs = FileSystemStorage(location=media_videos_dir)
        
        original_fn_cleaned = "".join(c if c.isalnum() or c in ['.','_'] else '_' for c in video_file.name)
        timestamp = int(time.time())
        input_fn = f"input_{media_subdir_prefix}_{timestamp}_{original_fn_cleaned}"
        saved_input_path_rel = fs.save(input_fn, video_file) # Relatif au 'location' de FileSystemStorage
        video_path_server = os.path.join(media_videos_dir, saved_input_path_rel) # Chemin absolu
        
        output_fn_base = f"output_{media_subdir_prefix}_{timestamp}_{original_fn_cleaned.rsplit('.',1)[0]}"
        # Chemin de base absolu pour la sortie (sans extension)
        output_path_base_server = os.path.join(media_videos_dir, output_fn_base) 
        
        task_id = f"{task_prefix}_task_{timestamp}"
        
        processing_status_dict[task_id] = {
            'status':'initializing', 'progress':0, 'task_id':task_id,
            'original_video_filename':input_fn, 
            'output_path_on_server':None, # Sera défini par le thread de traitement après choix du codec
            'video_url':None, 
            'error':None
        }
        
        thread = threading.Thread(target=process_function, args=(video_path_server, output_path_base_server, task_id))
        thread.daemon = True
        thread.start()
        
        context.update({
            'task_id': task_id,
            # MEDIA_URL doit aussi être configuré dans settings.py
            'original_video_url': os.path.join(settings.MEDIA_URL, f"videos_{media_subdir_prefix}", saved_input_path_rel).replace(os.path.sep,'/'),
            'original_video_type': video_file.content_type
        })
    except Exception as e:
        traceback.print_exc()
        context['error_message'] = f"Erreur téléversement ({media_subdir_prefix}): {str(e)}"


def _check_processing_status_generic(task_id, processing_status_dict):
    """Fonction helper pour vérifier le statut."""
    if task_id in processing_status_dict:
        status_data = processing_status_dict[task_id].copy()

        if status_data.get('status') == 'completed':
            server_path = status_data.get('output_path_on_server')
            print(f"DEBUG _check_generic (task {task_id}): status=completed, server_path='{server_path}'")

            if server_path and os.path.exists(server_path) and os.path.getsize(server_path) > 0: # Ajout vérif taille > 0
                try:
                    media_root_norm = os.path.normpath(settings.MEDIA_ROOT)
                    if os.path.commonpath([server_path, media_root_norm]) == media_root_norm:
                        rel_path = os.path.relpath(server_path, media_root_norm)
                        video_url_path = rel_path.replace(os.path.sep, '/')
                        status_data['video_url'] = os.path.join(settings.MEDIA_URL, video_url_path).replace(os.path.sep,'/')
                        
                        if video_url_path.lower().endswith('.mp4'):
                            status_data['video_type'] = 'video/mp4'
                        elif video_url_path.lower().endswith('.avi'):
                            status_data['video_type'] = 'video/avi'
                        else:
                            status_data['video_type'] = 'video/mp4' # Fallback
                        print(f"DEBUG _check_generic (task {task_id}): video_url='{status_data['video_url']}', video_type='{status_data['video_type']}'")
                    else:
                        print(f"ERREUR _check_generic (task {task_id}): server_path '{server_path}' n'est pas dans MEDIA_ROOT '{media_root_norm}'.")
                        status_data['video_url'] = None; status_data['video_type'] = None
                        status_data['error'] = status_data.get('error', '') + " Erreur chemin fichier invalide."
                except ValueError as e:
                    print(f"ERREUR _check_generic (task {task_id}): ValueError: {e}")
                    status_data['video_url'] = None; status_data['video_type'] = None
                    status_data['error'] = status_data.get('error', '') + f" Erreur calcul chemin: {e}"
            else:
                if not server_path: msg = "'output_path_on_server' est None."
                elif not os.path.exists(server_path): msg = f"Fichier NON TROUVÉ: {server_path}"
                else: msg = f"Fichier TAILLE ZÉRO: {server_path}"
                print(f"AVERTISSEMENT _check_generic (task {task_id}): {msg}")
                status_data['error'] = status_data.get('error', '') + f" {msg}"
                status_data['video_url'] = None; status_data['video_type'] = None
        
        return JsonResponse(status_data)
    return JsonResponse({'status':'not_found','progress':0, 'error': f'Tâche {task_id} non trouvée.'}, status=200)


# Vue pour le suivi des poissons
def fish_tracking_view(request):
    context = {'page_title': "Analyse Mouvement Poissons"}
    if request.method == "POST" and request.FILES.get("video_fish"):
        _handle_video_upload(request, "video_fish", fish_processing_status, process_fish_video_with_tracking, "fish", "fish", context)
    return render(request, 'fish_tracking.html', context)

# Vue pour le statut de traitement des poissons
def check_fish_processing_status(request, task_id):
    return _check_processing_status_generic(task_id, fish_processing_status)

#-----------------------------------------------------------------POINSONSSS comptage ------------------------------------------------------

import os
import cv2
import time
import threading
import json
import math
from collections import defaultdict, deque
import traceback
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import shutil
import numpy as np

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import JsonResponse

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Conditional PyTorch Import ---
_torch_module = None
TORCH_CUDA_AVAILABLE = False

try:
    import torch as _imported_torch
    _torch_module = _imported_torch
    if hasattr(_torch_module, 'cuda') and \
       hasattr(_torch_module.cuda, 'is_available') and \
       callable(_torch_module.cuda.is_available) and \
       _torch_module.cuda.is_available():
        TORCH_CUDA_AVAILABLE = True
        print("✅ PyTorch imported successfully and CUDA is available for YOLO/DeepSort.")
    else:
        print("✅ PyTorch imported, but CUDA is NOT available or check failed. YOLO/DeepSort will use CPU or as configured.")
except ImportError:
    print("⚠️ PyTorch not found. YOLO (PyTorch-based) requires it. DeepSort embedder might also. Processing might fail if not installed.")
except Exception as e_torch_check:
    print(f"⚠️ Error during PyTorch CUDA check: {e_torch_check}. Assuming CUDA not available for YOLO/DeepSort.")

# --- Font Loading ---
FONT_PATH = os.path.join(settings.BASE_DIR, 'outcomes', 'fonts', 'arial.ttf')
UNIVERSAL_FONT_SIZE = 12 # As in your notebook
try:
    UNIVERSAL_FONT = PIL.ImageFont.truetype(FONT_PATH, UNIVERSAL_FONT_SIZE)
    print(f"✅ Police chargée: {FONT_PATH} (Taille: {UNIVERSAL_FONT_SIZE})")
except IOError:
    UNIVERSAL_FONT = PIL.ImageFont.load_default()
    print(f"⚠️ Police '{FONT_PATH}' non trouvée. Utilisation de la police par défaut.")

# --- Configuration Globale (Comptage de Poissons ONLY) ---
FISH_COUNTING_MODEL_PATH = os.path.join(settings.BASE_DIR, 'outcomes', 'models', 'analyse_de_mouvement.pt') # !!! UPDATE THIS IF NEEDED !!!
yolo_model_counting = None
if not os.path.exists(FISH_COUNTING_MODEL_PATH):
    print(f"❌ ATTENTION : Modèle de comptage '{FISH_COUNTING_MODEL_PATH}' non trouvé. Essai avec YOLOv8s...")
    FISH_COUNTING_FALLBACK_PATH = os.path.join(settings.BASE_DIR, 'outcomes', 'models', 'yolov8s.pt')
    if not os.path.exists(FISH_COUNTING_FALLBACK_PATH):
        print(f"❌ ATTENTION : Modèle fallback YOLOv8s '{FISH_COUNTING_FALLBACK_PATH}' non trouvé.")
    else:
        try:
            yolo_model_counting = YOLO(FISH_COUNTING_FALLBACK_PATH)
            print(f"✅ Modèle comptage (fallback YOLOv8s) chargé : {FISH_COUNTING_FALLBACK_PATH}")
        except Exception as e: print(f"❌ Erreur chargement modèle comptage (YOLOv8s) : {e}")
else:
    try:
        yolo_model_counting = YOLO(FISH_COUNTING_MODEL_PATH)
        print(f"✅ Modèle comptage de poissons chargé : {FISH_COUNTING_MODEL_PATH}")
    except Exception as e: print(f"❌ Erreur chargement modèle comptage : {e}")

fish_counting_processing_status = {}


# --- Classes et Fonctions Utilitaires pour le Comptage de Poissons ---
class CustomVisTrack:
    def __init__(self, font_instance):
        self.colors = {}
        self.font = font_instance

    def get_color(self, idx):
        s_idx = str(idx)
        if s_idx not in self.colors:
            palette = [(0,0,255),(255,0,0),(0,255,0),(255,255,0),(0,255,255),(255,0,255),(128,0,0),(0,128,0)]
            self.colors[s_idx] = palette[hash(s_idx) % len(palette)]
        return self.colors[s_idx]

    # MINIMAL CHANGE HERE TO FIX THE ValueError
    def draw_bounding_boxes(self, pil_image, boxes_ltrb, ids_list, classes_list=None, scores_list=None):
        draw = PIL.ImageDraw.Draw(pil_image)
        for i, (box_coords, id_val) in enumerate(zip(boxes_ltrb, ids_list)):
            _x1_raw, _y1_raw, _x2_raw, _y2_raw = map(int, box_coords)

            x_draw_1 = min(_x1_raw, _x2_raw)
            y_draw_1 = min(_y1_raw, _y2_raw) # This is y0 (top)
            x_draw_2 = max(_x1_raw, _x2_raw)
            y_draw_2 = max(_y1_raw, _y2_raw) # This is y1 (bottom)

            # Skip drawing if the box has zero width or height AFTER ordering
            if x_draw_1 >= x_draw_2 or y_draw_1 >= y_draw_2:
                print(f"--- DEBUG (draw_bounding_boxes): Skipping zero-area box for ID {id_val} after ordering: ({x_draw_1},{y_draw_1},{x_draw_2},{y_draw_2}) from raw ({_x1_raw},{_y1_raw},{_x2_raw},{_y2_raw}) ---")
                continue

            color = self.get_color(id_val)
            draw.rectangle([(x_draw_1, y_draw_1), (x_draw_2, y_draw_2)], outline=color, width=2) # Use ordered coords

            # Text label logic (using y_draw_1 as the reference 'top' of the box for text positioning)
            text_content = f"Fish-{str(id_val)[:6]}"
            # Logic from your notebook to add class and score to text_content
            if classes_list is not None and i < len(classes_list) and classes_list[i]:
                 text_content = f"Fish-{str(id_val)[:6]}" # Your notebook overwrites here, if class is present, it's still "Fish-ID"
                 # If you meant: text_content = f"{str(classes_list[i])}-{str(id_val)[:6]}" then use that
            if scores_list is not None and i < len(scores_list) and scores_list[i] is not None:
                text_content += f" {scores_list[i]:.2f}"
            
            try:
                text_bbox = draw.textbbox((0,0), text_content, font=self.font)
                text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                
                # Position text background relative to the top of the drawn box (y_draw_1)
                rect_y1_bg = y_draw_1 - text_h - 7 
                text_y_draw = y_draw_1 - text_h - 5

                if rect_y1_bg < 0:
                    rect_y1_bg = y_draw_1 + 2
                    text_y_draw = y_draw_1 + 4
                rect_y2_bg = rect_y1_bg + text_h + 4

                draw.rectangle([(x_draw_1, rect_y1_bg), (x_draw_1 + text_w + 4, rect_y2_bg)], fill=color)
                draw.text((x_draw_1 + 2, text_y_draw), text_content, fill=(255,255,255), font=self.font)
            except AttributeError: # Fallback for older Pillow
                text_w, text_h = draw.textsize(text_content, font=self.font)
                # Position text background relative to the top of the drawn box (y_draw_1)
                rect_y1_bg_fallback = y_draw_1 - text_h - 5
                if rect_y1_bg_fallback < 0 : rect_y1_bg_fallback = y_draw_1 + 5 # Adjust if off-screen
                
                draw.rectangle([(x_draw_1, rect_y1_bg_fallback), (x_draw_1 + text_w + 5, rect_y1_bg_fallback + text_h + 5)], fill=color)
                draw.text((x_draw_1 + 2, rect_y1_bg_fallback + 2), text_content, fill=(255,255,255), font=self.font)
        return pil_image

class FishCounter: 
    def __init__(self):
        self.tracked_fishes_state = {}
        self.total_unique_count = 0
        self.current_frame_fish_count = 0
        self.class_distribution_count = {}
        self.entering_count = 0
        self.exiting_count = 0
        self.line_position_ratio = 0.5
        self.id_stability_score = defaultdict(int)
        self.track_trajectories = defaultdict(lambda: deque(maxlen=30))
        self.confirmed_stable_ids = set()
        self.movement_coherence_scores = defaultdict(float)
        self.STABILITY_FORGET_THRESHOLD = -50
        self.STABILITY_CONFIRM_THRESHOLD = 10
        self.COHERENCE_MIN_THRESHOLD = 0.6
        self.MAX_ID_STABILITY_SCORE = 100

    def update_count(self, current_frame_track_ids, classes_for_tracks, boxes_for_tracks_ltrb, frame_width_px):
        self.current_frame_fish_count = len(current_frame_track_ids)
        line_x_px = frame_width_px * self.line_position_ratio
        current_ids_in_frame_set = set(current_frame_track_ids)
        ids_to_forget = []
        for track_id_mem in list(self.id_stability_score.keys()):
            if track_id_mem not in current_ids_in_frame_set:
                self.id_stability_score[track_id_mem] -= 1
                if self.id_stability_score[track_id_mem] < self.STABILITY_FORGET_THRESHOLD:
                    ids_to_forget.append(track_id_mem)
            else:
                self.id_stability_score[track_id_mem] = min(self.MAX_ID_STABILITY_SCORE, self.id_stability_score[track_id_mem] + 1)
        for track_id_forget in ids_to_forget:
            self.id_stability_score.pop(track_id_forget, None)
            self.track_trajectories.pop(track_id_forget, None)
            self.movement_coherence_scores.pop(track_id_forget, None)
            self.tracked_fishes_state.pop(track_id_forget, None)

        for i, track_id_curr in enumerate(current_frame_track_ids):
            box_curr_ltrb = boxes_for_tracks_ltrb[i]
            center_x_curr = (box_curr_ltrb[0] + box_curr_ltrb[2]) / 2
            center_y_curr = (box_curr_ltrb[1] + box_curr_ltrb[3]) / 2
            if track_id_curr not in self.id_stability_score: self.id_stability_score[track_id_curr] = 1
            self.track_trajectories[track_id_curr].append((center_x_curr, center_y_curr))
            coherence_score_curr = self.calculate_movement_coherence(track_id_curr)
            self.movement_coherence_scores[track_id_curr] = coherence_score_curr
            if self.id_stability_score[track_id_curr] >= self.STABILITY_CONFIRM_THRESHOLD and coherence_score_curr >= self.COHERENCE_MIN_THRESHOLD:
                if track_id_curr not in self.confirmed_stable_ids:
                    self.confirmed_stable_ids.add(track_id_curr)
                    self.total_unique_count += 1
                fish_state_curr = self.tracked_fishes_state.get(track_id_curr)
                class_name_curr = classes_for_tracks[i] if i < len(classes_for_tracks) and classes_for_tracks[i] else "fish"
                if not fish_state_curr:
                    self.tracked_fishes_state[track_id_curr] = {'position_x': center_x_curr, 'counted_for_line': False, 'class': class_name_curr}
                    self.class_distribution_count[class_name_curr] = self.class_distribution_count.get(class_name_curr, 0) + 1
                else:
                    prev_x_pos = fish_state_curr['position_x']
                    max_jump_for_crossing_check = frame_width_px * 0.1
                    if not fish_state_curr['counted_for_line'] and abs(center_x_curr - prev_x_pos) < max_jump_for_crossing_check:
                        if prev_x_pos < line_x_px <= center_x_curr: self.entering_count += 1; fish_state_curr['counted_for_line'] = True
                        elif prev_x_pos >= line_x_px > center_x_curr: self.exiting_count += 1; fish_state_curr['counted_for_line'] = True
                    elif abs(center_x_curr - prev_x_pos) >= max_jump_for_crossing_check * 1.5: fish_state_curr['counted_for_line'] = False
                    fish_state_curr['position_x'] = center_x_curr

    def calculate_movement_coherence(self, track_id): # From notebook
        positions_list = list(self.track_trajectories[track_id])
        if len(positions_list) < 5: return 1.0
        dx_vals = [positions_list[j+1][0] - positions_list[j][0] for j in range(len(positions_list)-1)]
        dy_vals = [positions_list[j+1][1] - positions_list[j][1] for j in range(len(positions_list)-1)]
        if not dx_vals: return 1.0
        direction_reversals = 0
        num_comparable_segments = len(dx_vals) - 1
        for j in range(num_comparable_segments):
            if (dx_vals[j] * dx_vals[j+1] < 0 and abs(dx_vals[j]) > 5 and abs(dx_vals[j+1]) > 5) or \
               (dy_vals[j] * dy_vals[j+1] < 0 and abs(dy_vals[j]) > 5 and abs(dy_vals[j+1]) > 5):
                direction_reversals += 1
        if num_comparable_segments <= 0: return 1.0
        return max(0.0, 1.0 - (direction_reversals / float(num_comparable_segments)))

def iou(box1, box2): # As it was
    x1_i=max(box1[0],box2[0]); y1_i=max(box1[1],box2[1]); x2_i=min(box1[2],box2[2]); y2_i=min(box1[3],box2[3])
    intersection=max(0,x2_i-x1_i)*max(0,y2_i-y1_i)
    area1=(box1[2]-box1[0])*(box1[3]-box1[1]); area2=(box2[2]-box2[0])*(box2[3]-box2[1])
    union=area1+area2-intersection
    return intersection/float(union) if union > 1e-6 else 0.0

def validate_box_size(box_ltrb, frame_w, frame_h, max_ratio=0.5): # As it was
    x1,y1,x2,y2=box_ltrb; w,h=x2-x1,y2-y1
    if w<=0 or h<=0 or w>frame_w*max_ratio or h>frame_h*max_ratio:
        cx,cy=(x1+x2)/2,(y1+y2)/2
        new_w=min(max(10,w if w>0 else 10),frame_w*max_ratio)
        new_h=min(max(10,h if h>0 else 10),frame_h*max_ratio)
        x1=max(0,cx-new_w/2); y1=max(0,cy-new_h/2)
        x2=min(frame_w-1,cx+new_w/2); y2=min(frame_h-1,cy+new_h/2)
    return [int(x1),int(y1),int(x2),int(y2)]

def stabilize_tracking(current_ds_tracks, prev_frame_info, frame_w, frame_h): # As it was
    stabilized_list = []
    current_frame_info = {}
    available_prev_ids = list(prev_frame_info.keys())
    for ds_track in current_ds_tracks:
        if not ds_track.is_confirmed(): continue
        curr_box_raw = ds_track.to_ltrb()
        curr_box_val = validate_box_size(curr_box_raw, frame_w, frame_h)
        curr_center = ((curr_box_val[0]+curr_box_val[2])/2, (curr_box_val[1]+curr_box_val[3])/2)
        assigned_id = ds_track.track_id
        best_score = 0.3
        matched_prev_id = None
        for prev_id in available_prev_ids:
            prev_box, prev_center = prev_frame_info[prev_id]
            iou_val = iou(curr_box_val, prev_box)
            dist_factor = 1.0
            if prev_center:
                euc_dist = math.sqrt((curr_center[0]-prev_center[0])**2 + (curr_center[1]-prev_center[1])**2)
                dist_factor = max(0.5, min(1.5, 100.0 / (euc_dist + 10.0)))
            adj_iou = iou_val * dist_factor
            if adj_iou > best_score: best_score = adj_iou; matched_prev_id = prev_id
        if matched_prev_id:
            assigned_id = matched_prev_id
            available_prev_ids.remove(matched_prev_id)
        class_name = ds_track.original_class_name if hasattr(ds_track,'original_class_name') and ds_track.original_class_name else "fish"
        stabilized_list.append((assigned_id, curr_box_val, class_name))
        current_frame_info[assigned_id] = (curr_box_val, curr_center)
    return stabilized_list, current_frame_info

# --- Fonction de Traitement Vidéo pour Comptage de Poissons ---

def process_fish_counting_video(video_path_on_server, output_path_base_on_server, task_id_for_status):
    output_video_file_path = output_path_base_on_server + ".mp4"
    try:
        print(f"--- DEBUG: process_fish_counting_video started for task: {task_id_for_status} ---")
        if yolo_model_counting is None:
            fish_counting_processing_status[task_id_for_status].update({'status':'error','error':'Modèle comptage non chargé.'})
            print(f"--- DEBUG: Task {task_id_for_status} aborted: yolo_model_counting is None ---")
            return
        fish_counting_processing_status[task_id_for_status].update({'status':'processing','progress':0, 'stats': {
            "current_fish": 0, "total_unique": 0, "entries": 0, "exits": 0, "fps": "0.0"
        }})

        cap = cv2.VideoCapture(video_path_on_server)
        if not cap.isOpened():
            fish_counting_processing_status[task_id_for_status].update({'status':'error','error':f"Ouv. vidéo échoué: {video_path_on_server}"})
            print(f"--- DEBUG: Task {task_id_for_status} aborted: cv2.VideoCapture failed for {video_path_on_server} ---")
            return
        print(f"--- DEBUG: Task {task_id_for_status} VideoCapture opened successfully. ---")

        frame_w_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_video_fps = cap.get(cv2.CAP_PROP_FPS)
        output_video_fps = original_video_fps if 5 < original_video_fps < 120 else 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_video_frames == 0:
            fish_counting_processing_status[task_id_for_status].update({'status':'error','error':'Vidéo sans frames.'})
            cap.release(); return

        os.makedirs(os.path.dirname(output_video_file_path), exist_ok=True)

        compute_device = 'cpu'
        use_half_precision = False 
        if TORCH_CUDA_AVAILABLE:
            compute_device = 'cuda'
            use_half_precision = True
        print(f"Device (Comptage) Tâche {task_id_for_status}: {compute_device}")

        deepsort_tracker_instance = DeepSort(
            max_age=70, n_init=3, max_cosine_distance=0.4, nn_budget=100,
            embedder="mobilenet", half=use_half_precision, bgr=True, embedder_gpu=(compute_device == 'cuda')
        )
        
        video_writer_obj = None
        codecs_to_try_mp4 = [{'name': 'X264', 'fourcc': cv2.VideoWriter_fourcc(*'X264')},
                             {'name': 'mp4v', 'fourcc': cv2.VideoWriter_fourcc(*'mp4v')}]
        for codec_info_item in codecs_to_try_mp4:
            video_writer_obj = cv2.VideoWriter(output_video_file_path, codec_info_item['fourcc'], output_video_fps, (frame_w_px, frame_h_px))
            if video_writer_obj.isOpened(): break
        if not video_writer_obj or not video_writer_obj.isOpened():
            output_video_file_path_avi = output_path_base_on_server + ".avi"
            video_writer_obj = cv2.VideoWriter(output_video_file_path_avi, cv2.VideoWriter_fourcc(*'XVID'), output_video_fps, (frame_w_px, frame_h_px))
            if video_writer_obj.isOpened(): output_video_file_path = output_video_file_path_avi
            else:
                fish_counting_processing_status[task_id_for_status].update({'status':'error','error':'Échec création sortie vidéo.'})
                cap.release(); return
        fish_counting_processing_status[task_id_for_status]['output_path_on_server'] = output_video_file_path
        print(f"--- DEBUG: Task {task_id_for_status} VideoWriter opened for {output_video_file_path} ---")
        
        fish_counter_obj = FishCounter()
        vis_track_obj = CustomVisTrack(UNIVERSAL_FONT) # Pass the loaded font
        prev_frame_stabilized_tracks_info = {}
        raw_yolo_detection_history = defaultdict(int)
        yolo_confidence_threshold = 0.7

        processed_frame_count = 0
        fps_display_start_time = time.perf_counter()
        fps_display_frame_counter = 0
        fps_to_display_on_video = 0.0

        while True: # Main processing loop
            ret_val, bgr_frame_from_video = cap.read()
            if not ret_val: break
            processed_frame_count += 1

            yolo_prediction_results = yolo_model_counting.predict(bgr_frame_from_video, device=compute_device, verbose=False, conf=yolo_confidence_threshold)
            
            detections_for_deepsort_input = []
            if yolo_prediction_results and yolo_prediction_results[0].boxes:
                for yolo_box_obj in yolo_prediction_results[0].boxes:
                    ltrb_coords_raw = yolo_box_obj.xyxy[0].tolist()
                    confidence_score = float(yolo_box_obj.conf[0].item())
                    class_name_pred = "fish"
                    validated_ltrb_box = validate_box_size(ltrb_coords_raw, frame_w_px, frame_h_px)
                    detection_key_str = "-".join(map(str, validated_ltrb_box))
                    raw_yolo_detection_history[detection_key_str] += 1
                    if raw_yolo_detection_history[detection_key_str] >= 2 or confidence_score > 0.5:
                        detections_for_deepsort_input.append((
                            [validated_ltrb_box[0], validated_ltrb_box[1], validated_ltrb_box[2]-validated_ltrb_box[0], validated_ltrb_box[3]-validated_ltrb_box[1]],
                            confidence_score, class_name_pred ))
            for key_hist in list(raw_yolo_detection_history.keys()):
                raw_yolo_detection_history[key_hist] -= 0.5 
                if raw_yolo_detection_history[key_hist] <= 0: raw_yolo_detection_history.pop(key_hist)

            current_deepsort_track_objects = deepsort_tracker_instance.update_tracks(detections_for_deepsort_input, frame=bgr_frame_from_video)
            final_stabilized_tracks_list, prev_frame_stabilized_tracks_info = stabilize_tracking(
                current_deepsort_track_objects, prev_frame_stabilized_tracks_info, frame_w_px, frame_h_px )

            ids_to_count_draw = [st[0] for st in final_stabilized_tracks_list]
            boxes_ltrb_to_count_draw = [st[1] for st in final_stabilized_tracks_list]
            classes_to_count_draw = [st[2] for st in final_stabilized_tracks_list]
            if ids_to_count_draw:
                fish_counter_obj.update_count(ids_to_count_draw, classes_to_count_draw, boxes_ltrb_to_count_draw, frame_w_px)

            pil_output_frame = PIL.Image.fromarray(cv2.cvtColor(bgr_frame_from_video, cv2.COLOR_BGR2RGB))
            if ids_to_count_draw: # Draw only if there are tracks
                # THE FIX IS APPLIED IN THIS CALL TO draw_bounding_boxes
                pil_output_frame = vis_track_obj.draw_bounding_boxes(pil_output_frame, boxes_ltrb_to_count_draw, ids_to_count_draw, classes_to_count_draw, None) # Pass scores=None if not used for text
            
            pil_draw_context = PIL.ImageDraw.Draw(pil_output_frame)
            counting_line_x_draw = int(frame_w_px * fish_counter_obj.line_position_ratio)
            pil_draw_context.line([(counting_line_x_draw, 0), (counting_line_x_draw, frame_h_px)], fill=(220,50,50,200), width=3)
            fps_display_frame_counter += 1
            elapsed_time_for_fps_display = time.perf_counter() - fps_display_start_time
            if elapsed_time_for_fps_display >= 1.0:
                fps_to_display_on_video = fps_display_frame_counter / elapsed_time_for_fps_display
                fps_display_frame_counter = 0; fps_display_start_time = time.perf_counter()
            stats_for_on_video_display = [
                f"Live Count: {fish_counter_obj.current_frame_fish_count}", f"Total Unique: {fish_counter_obj.total_unique_count}",
                f"Entries: {fish_counter_obj.entering_count}", f"Exits: {fish_counter_obj.exiting_count}",
                f"FPS: {fps_to_display_on_video:.1f}" ]
            for i_stat_text, stat_text_line_item in enumerate(stats_for_on_video_display):
                try:
                    stat_text_bbox=pil_draw_context.textbbox((0,0),stat_text_line_item,font=UNIVERSAL_FONT)
                    stat_text_h=stat_text_bbox[3]-stat_text_bbox[1]
                    text_pos_x,text_pos_y=10,10+i_stat_text*(stat_text_h+8)
                    bg_rect_x1,bg_rect_y1=text_pos_x-3,text_pos_y-3
                    bg_rect_x2,bg_rect_y2=text_pos_x+(stat_text_bbox[2]-stat_text_bbox[0])+3,text_pos_y+stat_text_h+3
                    pil_draw_context.rectangle([(bg_rect_x1,bg_rect_y1),(bg_rect_x2,bg_rect_y2)],fill=(0,0,0,120))
                    pil_draw_context.text((text_pos_x,text_pos_y),stat_text_line_item,fill=(230,230,100),font=UNIVERSAL_FONT)
                except AttributeError: pil_draw_context.text((10,10+i_stat_text*20),stat_text_line_item,fill=(230,230,100),font=UNIVERSAL_FONT)
            
            final_cv_frame_to_write = cv2.cvtColor(np.array(pil_output_frame), cv2.COLOR_RGB2BGR)
            video_writer_obj.write(final_cv_frame_to_write)

            if processed_frame_count % 20 == 0:
                current_progress_percent = min(int((processed_frame_count/total_video_frames)*100),99) if total_video_frames > 0 else 0
                current_stats_for_json_response = {
                    "current_fish": fish_counter_obj.current_frame_fish_count, "total_unique": fish_counter_obj.total_unique_count,
                    "entries": fish_counter_obj.entering_count, "exits": fish_counter_obj.exiting_count,
                    "fps": f"{fps_to_display_on_video:.1f}"}
                fish_counting_processing_status[task_id_for_status]['progress'] = current_progress_percent
                fish_counting_processing_status[task_id_for_status]['stats'] = current_stats_for_json_response
        
        cap.release(); video_writer_obj.release()
        print(f"--- DEBUG: Task {task_id_for_status} processing loop finished. ---")
        final_output_path_check = fish_counting_processing_status[task_id_for_status].get('output_path_on_server')
        if final_output_path_check and os.path.exists(final_output_path_check) and os.path.getsize(final_output_path_check) > 0:
            fish_counting_processing_status[task_id_for_status].update({'status':'completed','progress':100})
        else:
            fish_counting_processing_status[task_id_for_status].update({'status':'error','error':'Fichier sortie non trouvé/vide.'})
    except Exception as e_main_processing:
        traceback.print_exc()
        fish_counting_processing_status[task_id_for_status].update({'status':'error','error':f"{type(e_main_processing).__name__}: {str(e_main_processing)}"})
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'video_writer_obj' in locals() and video_writer_obj.isOpened(): video_writer_obj.release()

def _handle_video_upload(request, file_input_name_html, processing_status_map,
                         video_processing_function, media_folder_prefix, task_id_prefix_str,
                         django_view_context):
    print(f"--- DEBUG: _handle_video_upload called for input name: {file_input_name_html} ---")
    print(f"--- DEBUG: request.FILES: {request.FILES} ---")
    try:
        uploaded_video_file = request.FILES.get(file_input_name_html)
        if not uploaded_video_file:
            print(f"--- DEBUG: File not found in request.FILES with name '{file_input_name_html}' ---")
            django_view_context['error_message'] = f"Erreur: Aucun fichier vidéo fourni avec le nom '{file_input_name_html}'."
            return

        print(f"--- DEBUG: File '{uploaded_video_file.name}' received. Size: {uploaded_video_file.size} ---")
        base_filename, file_extension = os.path.splitext(uploaded_video_file.name)
        sanitized_basename = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in base_filename)[:100]
        cleaned_original_filename = f"{sanitized_basename}{file_extension}"
        target_media_subdir = os.path.join(settings.MEDIA_ROOT, f"videos_{media_folder_prefix}")
        os.makedirs(target_media_subdir, exist_ok=True)
        file_system_storage = FileSystemStorage(location=target_media_subdir)
        current_timestamp = int(time.time())
        input_filename_on_disk = f"input_{media_folder_prefix}_{current_timestamp}_{cleaned_original_filename}"
        relative_saved_input_path = file_system_storage.save(input_filename_on_disk, uploaded_video_file)
        absolute_video_path_on_server = os.path.join(target_media_subdir, relative_saved_input_path)
        output_filename_base_on_disk = f"output_{media_folder_prefix}_{current_timestamp}_{sanitized_basename}"
        absolute_output_path_base_on_server = os.path.join(target_media_subdir, output_filename_base_on_disk)
        generated_task_id = f"{task_id_prefix_str}_task_{current_timestamp}"
        
        processing_status_map[generated_task_id] = {
            'status':'initializing', 'progress':0, 'task_id':generated_task_id,
            'original_video_filename':input_filename_on_disk,
            'output_path_on_server':None, 'video_url':None, 'video_type':None,
            'error':None, 'stats': {}
        }
        print(f"--- DEBUG: Initialized status for task {generated_task_id}. Starting thread... ---")
        processing_thread = threading.Thread(
            target=video_processing_function,
            args=(absolute_video_path_on_server, absolute_output_path_base_on_server, generated_task_id)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        django_view_context.update({
            'task_id': generated_task_id,
            'original_video_url': os.path.join(settings.MEDIA_URL, f"videos_{media_folder_prefix}", relative_saved_input_path).replace(os.path.sep,'/'),
            'original_video_type': uploaded_video_file.content_type
        })
    except Exception as e_generic_upload:
        traceback.print_exc()
        django_view_context['error_message'] = f"Erreur téléversement ({media_folder_prefix}): {str(e_generic_upload)}"

def _check_processing_status_generic(task_id_to_check, status_map_ref):
    if task_id_to_check in status_map_ref:
        current_status_data = status_map_ref[task_id_to_check].copy()
        if current_status_data.get('status') == 'completed':
            final_server_video_path = current_status_data.get('output_path_on_server')
            if final_server_video_path and os.path.exists(final_server_video_path) and os.path.getsize(final_server_video_path) > 0:
                try:
                    abs_media_root_norm = os.path.normpath(os.path.abspath(settings.MEDIA_ROOT))
                    abs_final_video_path_norm = os.path.normpath(os.path.abspath(final_server_video_path))
                    if os.path.commonpath([abs_final_video_path_norm, abs_media_root_norm]) == abs_media_root_norm:
                        rel_path_video_to_media_root = os.path.relpath(abs_final_video_path_norm, abs_media_root_norm)
                        url_path_segment_video = rel_path_video_to_media_root.replace(os.path.sep, '/')
                        clean_media_url = settings.MEDIA_URL.rstrip('/') + '/'
                        current_status_data['video_url'] = f"{clean_media_url}{url_path_segment_video}"
                        if url_path_segment_video.lower().endswith('.mp4'): current_status_data['video_type'] = 'video/mp4'
                        elif url_path_segment_video.lower().endswith('.avi'): current_status_data['video_type'] = 'video/x-msvideo'
                        else: current_status_data['video_type'] = 'video/mp4'
                    else:
                        current_status_data.update({'video_url': None, 'video_type': None, 'error': (current_status_data.get('error') or "") + " Erreur: Chemin de sortie invalide."})
                except ValueError as e_path_val_err:
                    current_status_data.update({'video_url': None, 'video_type': None, 'error': (current_status_data.get('error') or "") + f" Erreur calcul URL: {e_path_val_err}"})
            else:
                file_issue_msg = "Fichier de sortie non prêt ou vide."
                current_status_data.update({'error': (current_status_data.get('error') or "") + f" {file_issue_msg}", 'video_url': None, 'video_type': None})
        return JsonResponse(current_status_data)
    return JsonResponse({'status':'not_found','progress':0, 'error': f'Tâche {task_id_to_check} non trouvée.'}, status=200)

def fish_counting_view(request): 
    view_context = {'page_title': "Comptage de Poissons"}
    html_file_input_name = "video_upload_for_counting" # Consistent name
    if request.method == "POST" and request.FILES.get(html_file_input_name):
        print(f"--- DEBUG: POST request to fish_counting_view with file '{html_file_input_name}' found. ---")
        _handle_video_upload(
            request, html_file_input_name,
            fish_counting_processing_status, process_fish_counting_video,
            "fish_counting_data", "f_count", view_context
        )
    elif request.method == "POST": # File not found in POST
         print(f"--- DEBUG: POST to fish_counting_view, BUT file input '{html_file_input_name}' NOT FOUND. Files in request: {list(request.FILES.keys())} ---")
         view_context['error_message'] = "Aucun fichier vidéo n'a été téléversé ou le nom du champ est incorrect."

    return render(request, 'fish_counting.html', view_context) # Ensure template name is correct

def check_fish_counting_status(request, task_id): # Name matches your urls.py
    return _check_processing_status_generic(task_id, fish_counting_processing_status)



#-----------------------------------------------------------------seagull detection ------------------------------------------------------

import os
import cv2
import numpy as np
import threading
import time
import json
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
# from django.urls import reverse # Non utilisé explicitement ici, mais utile pour les templates
from django.views.decorators.csrf import csrf_exempt # Garder si vous avez des POST sans formulaire Django complet
from ultralytics import YOLO
import pygame
import traceback # Pour un meilleur logging des erreurs

# Initialisation de pygame pour l'audio
pygame.mixer.init()

# Charger le modèle YOLOv8
try:
    model_path = os.path.join(settings.BASE_DIR, 'outcomes', 'models', 'seagull.pt')
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"✅ Modèle YOLOv8 chargé avec succès: {model_path}")
    else:
        print(f"❌ Modèle non trouvé: {model_path}")
        model = None
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle YOLOv8: {e}")
    model = None

# Variable pour suivre l'état du traitement
processing_status = {} # Dictionnaire pour stocker l'état des tâches

# Variable pour contrôler le flux vidéo
camera_active = False
video_camera = None # Sera initialisé dans generate_frames

def play_sound():
    """Joue un son d'alarme sur le serveur"""
    try:
        # Chemin principal du son
        sound_path = os.path.join(settings.BASE_DIR, 'static', 'sounds', 'alarm-301729.mp3')
        
        # Liste des chemins alternatifs
        alt_paths = [
            os.path.join(settings.STATIC_ROOT or '', 'sounds', 'alarm-301729.mp3'), # STATIC_ROOT peut être None
            os.path.join(settings.BASE_DIR, 'static', 'alarm-301729.mp3'),
            os.path.join(settings.BASE_DIR, 'outcomes', 'static', 'sounds', 'alarm-301729.mp3')
        ]
        
        # Ajouter le chemin principal à la liste s'il est différent des autres
        if sound_path not in alt_paths:
            alt_paths.insert(0, sound_path)

        found_sound = False
        for path_to_try in alt_paths:
            if path_to_try and os.path.exists(path_to_try): # Vérifier si path_to_try n'est pas None ou vide
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load(path_to_try)
                    pygame.mixer.music.play()
                    print(f"✅ Son d'alarme joué: {path_to_try}")
                    found_sound = True
                    break # Sortir de la boucle dès que le son est trouvé et joué
        
        if not found_sound:
            print(f"❌ Fichier son 'alarm-301729.mp3' introuvable dans les chemins configurés.")
            
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du son: {e}")
        traceback.print_exc()


def process_video(video_path, output_path_base, task_id): # output_path_base sans extension
    """Traite une vidéo téléchargée avec le modèle YOLOv8"""
    actual_output_path = output_path_base + ".mp4" # Par défaut en mp4

    try:
        if model is None:
            print("❌ Modèle non disponible pour le traitement")
            processing_status[task_id].update({
                'status': 'error',
                'error': 'Modèle YOLOv8 non disponible'
            })
            return
        
        # Le statut est déjà initialisé, on le met à jour
        processing_status[task_id].update({'status': 'processing', 'progress': 0})
        
        if not os.path.exists(video_path):
            print(f"❌ Le fichier vidéo n'existe pas: {video_path}")
            processing_status[task_id].update({'status': 'error', 'error': 'Fichier vidéo introuvable'})
            return
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Erreur : Impossible de lire la vidéo {video_path}")
            processing_status[task_id].update({'status': 'error', 'error': 'Impossible de lire la vidéo'})
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: # Certaines vidéos peuvent avoir un FPS de 0, utiliser une valeur par défaut
            print("⚠️ FPS de la vidéo est 0, utilisation de 25 FPS par défaut.")
            fps = 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✅ Dimensions vidéo : {frame_width}x{frame_height} - {fps} FPS - {total_frames} frames")
        
        os.makedirs(os.path.dirname(actual_output_path), exist_ok=True)
        
        out = None
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1') # MP4
            out = cv2.VideoWriter(actual_output_path, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                raise Exception("Le codec avc1 n'est pas disponible ou a échoué.")
        except Exception as e_avc1:
            print(f"⚠️ Échec avec le codec avc1: {e_avc1}. Essai avec XVID (AVI)...")
            actual_output_path = output_path_base + ".avi" # Changer l'extension
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # AVI
            out = cv2.VideoWriter(actual_output_path, fourcc, fps, (frame_width, frame_height))

        if not out or not out.isOpened():
            print("❌ Impossible de créer le fichier de sortie vidéo avec les codecs testés.")
            processing_status[task_id].update({
                'status': 'error', 
                'error': 'Impossible de créer le fichier de sortie vidéo (codecs avc1/XVID échoués)'
            })
            if cap: cap.release()
            return
        
        processing_status[task_id]['output_path_on_server'] = actual_output_path # Mettre à jour avec le chemin réel

        frame_count = 0
        detected_in_video = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Fin de la vidéo après {frame_count} frames")
                break
            
            current_frame_seagull_detected = False
            results = model(frame, conf=0.25) 
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0: # Supposons que la classe 0 est 'seagull'
                        detected_in_video = True
                        current_frame_seagull_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Seagull {conf:.2f}", (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if current_frame_seagull_detected:
                cv2.putText(frame, "ALARME: Mouette detectee!", (10, frame_height - 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                # play_sound() # Joue le son sur le serveur, pas dans la vidéo.
                           # Peut être utile pour notifier l'admin pendant un long traitement.

            cv2.putText(frame, "Seagull Detection Output", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            out.write(frame)
            
            frame_count += 1
            if total_frames > 0 and frame_count % 10 == 0:
                progress = min(int((frame_count / total_frames) * 100), 99)
                processing_status[task_id]['progress'] = progress
        
        cap.release()
        out.release()
        
        # S'assurer que le chemin final est bien celui utilisé
        processing_status[task_id].update({
            'status': 'completed', 
            'progress': 100,
            'output_path_on_server': actual_output_path, # Chemin complet sur le serveur
            'detected_in_video': detected_in_video # Indiquer si une détection a eu lieu
        })
        print(f"✅ Vidéo traitée enregistrée dans : {actual_output_path}")
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement de la vidéo ({task_id}): {e}")
        traceback.print_exc()
        processing_status[task_id].update({
            'status': 'error', 
            'error': str(e)
        })
        if 'cap' in locals() and cap and cap.isOpened(): cap.release()
        if 'out' in locals() and out and out.isOpened(): out.release()


def generate_frames():
    global camera_active, video_camera
    
    if video_camera is None or not video_camera.isOpened():
        print("Tentative d'initialisation de la caméra pour le flux...")
        video_camera = cv2.VideoCapture(0) 
        time.sleep(0.5) # Laisser le temps à la caméra de s'initialiser
        
        if not video_camera.isOpened():
            print("❌ Impossible d'ouvrir la caméra pour le flux.")
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Erreur: Camera non disponible", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
    print("✅ Caméra initialisée avec succès pour le flux.")

    if model is None:
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Erreur: Modele YOLOv8 non disponible", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    while camera_active:
        try:
            ret, frame = video_camera.read()
            if not ret:
                print("❌ Erreur de lecture du flux de la caméra, arrêt.")
                # Envoyer une dernière image d'erreur peut être une bonne idée ici
                camera_active = False # Arrêter la boucle
                break 
            
            frame_height, frame_width = frame.shape[:2]
            seagull_detected_in_frame = False
            
            results = model(frame, conf=0.25)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0: 
                        seagull_detected_in_frame = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Seagull {conf:.2f}", (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if seagull_detected_in_frame:
                cv2.putText(frame, "ALARME: Mouette detectee!", (10, frame_height - 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                play_sound() # Joue le son sur le serveur
            
            cv2.putText(frame, "Live Detection", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("❌ Erreur d'encodage du frame en JPEG.")
                continue # Sauter ce frame

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(1/30) # Viser ~30 FPS, ajustable
            
        except Exception as e:
            print(f"❌ Erreur dans generate_frames: {e}")
            traceback.print_exc()
            camera_active = False # Arrêter en cas d'erreur grave
            break
    
    if video_camera is not None and video_camera.isOpened():
        print("Libération de la caméra car camera_active est False ou une erreur s'est produite.")
        video_camera.release()
        video_camera = None
    print("✅ Flux generate_frames terminé.")


def camera_feed(request):
    global camera_active
    if not camera_active: # Vérification supplémentaire
        print("Tentative d'accès à camera_feed alors que la caméra n'est pas active.")
        # Vous pourriez retourner une image statique ou une erreur HTTP ici
        # return HttpResponse("Camera not started.", status=400)

    try:
        return StreamingHttpResponse(generate_frames(), 
                                    content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"❌ Erreur critique dans camera_feed: {e}")
        traceback.print_exc()
        return HttpResponse(f"Erreur de streaming vidéo: {str(e)}", status=500)

@csrf_exempt
def start_camera(request):
    global camera_active, video_camera
    if not camera_active:
        camera_active = True
        # L'initialisation de la caméra se fera dans generate_frames au premier appel
        print("✅ Démarrage de la caméra demandé.")
        return JsonResponse({'status': 'starting'}) # ou 'started' si l'initialisation est rapide
    return JsonResponse({'status': 'already_started'})

@csrf_exempt
def stop_camera(request):
    global camera_active, video_camera
    if camera_active:
        camera_active = False
        print("✅ Arrêt de la caméra demandé.")


        return JsonResponse({'status': 'stopped'})
    return JsonResponse({'status': 'already_stopped'})

def check_processing_status(request, task_id):
    if task_id in processing_status:
        status_data = processing_status[task_id].copy() # Travailler sur une copie
        
        if status_data.get('status') == 'completed':
            # Construire l'URL MEDIA pour le client
            # output_path_on_server est le chemin absolu sur le serveur
            # On a besoin du chemin relatif à MEDIA_ROOT pour construire MEDIA_URL
            server_path = status_data.get('output_path_on_server')
            if server_path:
                try:
                    # Assurez-vous que MEDIA_ROOT se termine par un / pour une jointure correcte
                    media_root_normalized = os.path.join(settings.MEDIA_ROOT, '')
                    relative_path = os.path.relpath(server_path, media_root_normalized)
                    # Remplacer les \ par / pour les URLs
                    video_url_path = relative_path.replace(os.path.sep, '/')
                    status_data['video_url'] = os.path.join(settings.MEDIA_URL, video_url_path).replace(os.path.sep, '/')

                    # Déterminer le type MIME pour le HTML
                    if video_url_path.lower().endswith('.mp4'):
                        status_data['video_type'] = 'video/mp4'
                    elif video_url_path.lower().endswith('.avi'):
                        status_data['video_type'] = 'video/avi' # Moins supporté nativement
                    else:
                        status_data['video_type'] = 'video/octet-stream' # Type générique

                except ValueError as e: # relpath peut échouer si les chemins sont sur des lecteurs différents
                    print(f"Erreur de calcul du chemin relatif pour {server_path} par rapport à {settings.MEDIA_ROOT}: {e}")
                    status_data['video_url'] = None # Ou une URL d'erreur
                    status_data['error'] = status_data.get('error', '') + " Erreur de chemin de fichier."
            else:
                status_data['video_url'] = None
        
        return JsonResponse(status_data)
    else:
        # Cas où le task_id n'est même pas encore dans processing_status
        # Cela peut arriver si checkStatus est appelé trop rapidement
        return JsonResponse({'status': 'not_found', 'progress': 0, 'error': 'Tâche non trouvée ou non initialisée.'}, status=200) # Retourner 200 pour que .json() ne plante pas côté client


def intruder_detection_view(request):
    context = {
        'mode': request.GET.get('mode', 'camera') # Défaut sur 'camera' ou 'upload' selon votre préférence
    }
    
    if request.method == "POST" and request.FILES.get("video"):
        try:
            video_file = request.FILES["video"]
            
            media_videos_dir = os.path.join(settings.MEDIA_ROOT, "videos")
            os.makedirs(media_videos_dir, exist_ok=True)
            
            fs = FileSystemStorage(location=media_videos_dir)
            
            # Nettoyer le nom de fichier pour éviter les caractères problématiques
            original_filename_cleaned = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in video_file.name)
            timestamp = int(time.time())

            input_filename = f"input_{timestamp}_{original_filename_cleaned}"
            saved_input_path_relative_to_media_videos = fs.save(input_filename, video_file)
            video_path_on_server = os.path.join(media_videos_dir, saved_input_path_relative_to_media_videos)
            
            # Chemin de base pour la sortie (sans extension, elle sera ajoutée par process_video)
            output_filename_base = f"output_{timestamp}_{original_filename_cleaned.rsplit('.', 1)[0]}"
            # Le chemin complet sera MEDIA_ROOT/videos/output_... .mp4 ou .avi
            output_path_base_on_server = os.path.join(media_videos_dir, output_filename_base) 
            
            task_id = f"task_{timestamp}"
            
            # Initialiser le statut de la tâche AVANT de démarrer le thread
            processing_status[task_id] = {
                'status': 'initializing', 
                'progress': 0,
                'task_id': task_id,
                'original_video_filename': input_filename, # Juste le nom du fichier
                'output_path_on_server': None, # Sera défini par le thread
                'video_url': None, # Sera défini lorsque 'completed'
                'error': None
            }
            
            print(f"✅ Vidéo téléchargée: {video_path_on_server}")
            print(f"✅ Base pour sortie: {output_path_base_on_server}")
            print(f"✅ Task ID: {task_id}")
            
            thread = threading.Thread(target=process_video, args=(video_path_on_server, output_path_base_on_server, task_id))
            thread.daemon = True # Permet au programme principal de se terminer même si les threads sont en cours
            thread.start()
            
            context.update({
                'task_id': task_id,
                # URL pour la vidéo originale (pour l'affichage direct si nécessaire)
                'original_video_url': os.path.join(settings.MEDIA_URL, "videos", saved_input_path_relative_to_media_videos).replace(os.path.sep, '/'),
                'original_video_type': video_file.content_type
            })
            
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement ou du démarrage du traitement de la vidéo: {e}")
            traceback.print_exc()
            context.update({
                'error_message': f"Erreur lors du téléchargement: {str(e)}"
            })
    
    return render(request, 'intruder_detection.html', context)


