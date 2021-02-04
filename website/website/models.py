from django.db import models

# Create your models here.
class Image(models.Model):
    #title = models.TextField()
    image = models.ImageField(upload_to='images')   # MEDIA_ROOT/images/

    def __str__(self):
        return str(self.image)