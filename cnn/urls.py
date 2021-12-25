from django.urls import path
from cnn.views import Index,ModelTrain,Classifier
from . import views

app_name = 'cnn'

urlpatterns = [
    path('', Index.as_view(),name='index'),
    path('model',ModelTrain.as_view() ,name='model'),    
    path('classifier',Classifier.as_view(), name='classifier'),
    path('upload_datasets',views.upload_datasets, name='upload_datasets'),
    path('upload_classify_data', views.upload_classify_data, name='upload_classify_data'),

    path('train_model',views.train_model, name='train_model'),
    path('v_remove_train_file',views.v_remove_train_file,name='v_remove_file'),
    path('v_remove_classify_file', views.v_remove_classify_file, name='v_remove_file'),

    path('v_classify',views.v_classify,name='v_classify'),
    path('img_loss',views.v_img_loss, name='img_loss'),
    path('img_accuracy', views.v_img_accuracy, name='img_accuracy')
]

