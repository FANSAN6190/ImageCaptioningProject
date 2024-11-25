from django.shortcuts import render, redirect
from .models import Image
from .forms import ImageForm
from .captioning_model import generate_caption 

def upload_image(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save()
            # Here you would add the image captioning logic
            image.caption = generate_caption(image.image.path)
            image.save()
            return redirect('image_detail', pk=image.pk)
    else:
        form = ImageForm()
    return render(request, 'captioning/upload_image.html', {'form': form})

def image_detail(request, pk):
    image = Image.objects.get(pk=pk)
    return render(request, 'captioning/image_detail.html', {'image': image})