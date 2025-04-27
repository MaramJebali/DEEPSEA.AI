from django.shortcuts import render
from django.http import JsonResponse
from .marine_chatbot import get_response  
from django.views.decorators.csrf import csrf_exempt


def index(request):
    return render(request, 'index.html')


def chatbot_page(request):
    return render(request, 'chatbot.html')  # ➔ Your chatbot page

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
