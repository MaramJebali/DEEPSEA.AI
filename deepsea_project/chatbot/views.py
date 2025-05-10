from django.shortcuts import render
from django.http import JsonResponse
from .marine_chatbot import get_response  
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .shield_noise import load_audio, classify_audio, build_user_query, generate_report
import os

def index(request):
    return render(request, 'index.html')



def services(request):
    return render(request, 'getstarted.html')  # ➔ Your chatbot page

def chatbot_page(request):
    return render(request, 'chatbot.html')  # ➔ Your chatbot page

def marinehealth(request):
    return render(request, 'Marine_health.html')
def aqua(request):
    return render(request, 'aquaculture.html') 
def overfish(request):
    return render(request, 'overfishing.html') 

@csrf_exempt
def get_bot_response(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '')
        if user_message:
            bot_reply = get_response(user_message)
            return JsonResponse({'response': bot_reply})
        else:
            return JsonResponse({'response': "❌ Please enter a valid message."})
    else:
        return JsonResponse({'response': "❌ Invalid request method."})
def chatbot(request):
    return render(request, 'chatbot.html')



@csrf_exempt
def noise_analysis(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        location = request.POST.get('location', 'Unknown')
        date = request.POST.get('date', 'Unknown')
        species = request.POST.get('species', 'Unknown species').split(',')

        # Save audio file to media directory
        file_path = f'media/{audio_file.name}'
        with open(file_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        # Load and classify audio
        audio = load_audio(file_path)
        tags = classify_audio(audio)

        # Context for analysis
        context = {
            "location": location,
            "date": date,
            "present_species": [s.strip() for s in species]
        }

        # Build query and generate report
        user_query = build_user_query(tags, context)
        report = generate_report(user_query)

        # Return result as JSON
        return JsonResponse({"tags": tags, "report": report})

    return render(request, 'noiseshield.html')
