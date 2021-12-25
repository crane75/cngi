from django import forms


class ModelTrainForm(forms.Form):
    train_size = forms.FloatField(label='train_size: ', initial=0.8, max_value=0.99, min_value=0.01, widget=forms.TextInput)
    batch_size = forms.IntegerField(label='batch_size:', initial=512, max_value=2048, min_value=16, widget=forms.TextInput)
    stride = forms.IntegerField(label='stride:', initial=1, widget=forms.TextInput)
    lr = forms.FloatField(label='learn_rate:', initial=0.01, min_value=0.0001, max_value=0.5, widget=forms.TextInput)
    epoch = forms.IntegerField(label='epoch:', initial=10, min_value=1, widget=forms.TextInput)


class ClassifyForm(forms.Form):
    datafile = forms.FileField(label='文件名', widget=forms.FileInput)









