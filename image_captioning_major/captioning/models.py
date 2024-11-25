from django.db import models

class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    caption = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image {self.id}"