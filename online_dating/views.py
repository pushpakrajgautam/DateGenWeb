from django.http import HttpResponseRedirect
from django.shortcuts import render

from .forms import ProfileForm
from . import dp_gen_web as gen

def get_name(request):
    if request.method == 'POST':
        form = ProfileForm(request.POST)
        if form.is_valid():
            profile = gen.get_profile(request.POST)
            return render(request, 'index.html', {'form': form, 'output': profile})
    else:
        form = ProfileForm()
    return render(request, 'index.html', {'form': form})

gen.setup() # Load the LSTM model with weights