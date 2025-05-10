from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout, get_user_model
from django.contrib import messages

User = get_user_model()

def getstarted(request):
    if request.method == 'POST':
        mode = request.POST.get('mode')

        if mode == 'signup':
            username = request.POST.get('new-username')
            email = request.POST.get('email')
            password = request.POST.get('new-password')
            confirm_password = request.POST.get('confirm-password')

            if password != confirm_password:
                messages.error(request, "Passwords do not match.")
                return redirect('users:getstarted')

            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already exists.")
                return redirect('users:getstarted')

            if User.objects.filter(email=email).exists():
                messages.error(request, "Email already in use.")
                return redirect('users:getstarted')

            user = User.objects.create_user(username=username, email=email, password=password)
            login(request, user)
            return redirect('index')

        elif mode == 'login':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('index')
            else:
                messages.error(request, "Invalid username or password.")
                return redirect('users:getstarted')

    return render(request, 'getstarted.html')

def logout_view(request):
    logout(request)
    return redirect('index')
