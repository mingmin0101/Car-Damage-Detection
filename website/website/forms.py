from django import forms
from .models import Image


class ImageForm(forms.ModelForm):
    """Form for the image model"""
    image = forms.ImageField()

    class Meta:
        model = Image
        fields = ('image',)