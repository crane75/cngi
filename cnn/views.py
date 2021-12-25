from django.http import HttpResponse, HttpResponseRedirect
from django.views import generic
from django.contrib import messages
from .forms import ModelTrainForm, ClassifyForm
import os

from .cnn import train, remove_train_file, remove_classify_file,classfy


# Create your views here.

# Index Page
class Index(generic.base.TemplateView):
    template_name = 'cnn/index.html'


# Model Page
class ModelTrain(generic.FormView):
    template_name = 'cnn/modeltrain.html'
    form_class = ModelTrainForm

    def get_context_data(self, **kwargs):
        context = super(ModelTrain,self).get_context_data(**kwargs)
        context['filenames'] = os.listdir('cnn/static/datas')
        return context


class Classifier(generic.FormView):
    template_name = 'cnn/classify.html'
    form_class = ClassifyForm

    def get_context_data(self, **kwargs):
        context = super(Classifier,self).get_context_data(**kwargs)
        context['filenames'] = os.listdir('cnn/classify-data')
        return context

def upload(request, path, destfilename=None):
    if request.method == 'POST':
        datafile = request.FILES.get('datafile',)
        if not datafile:
            return -1       # 没有需要上传的文件
        if not (datafile.name.endswith(".pcapng") or datafile.name.endswith(".pcap")):
            return 1        # 请选择正确的文件类型
        if not destfilename:
            destination = open(os.path.join(path, datafile.name), 'wb+')
        else:
            destination = open(os.path.join(path, destfilename), 'wb+')
        for chunk in datafile.chunks():
            destination.write(chunk)
        destination.close()
        return 0


def upload_datasets(request):
    path = './cnn/static/datas/'
    state = upload(request, path)
    if state == -1:
        messages.error(request,'没有需要上传的文件')
    if state == 1:
        messages.error(request,'请选择正确的文件类型')
    return HttpResponseRedirect('model')


def upload_classify_data(request):
    path = './cnn/classify-data/'
    state = upload(request, path)
    if state == -1:
        messages.error(request,'没有需要上传的文件')
    if state == 1:
        messages.error(request,'请选择正确的文件类型')
    if state == 0:
        messages.error(request,"上传文件成功")
    return HttpResponseRedirect('classifier')


def train_model(request):
    trainsize = float(request.POST['train_size'])
    batch_size = int(request.POST['batch_size'])
    stride = int(request.POST['stride'])
    lr = float(request.POST['lr'])
    epoch = int(request.POST['epoch'])
    report = train(trainsize=trainsize,batchsize=batch_size,stride=stride,learn_rate=lr,epoch=epoch)
    message = '<table class="table table-striped"><tr>\n'
    message += '<th></th><th>precision</th>\n' + '<th>recall</th>\n' + '<th>f1-score</th>\n' + '<th>support</th></tt>\n'
    for instance in report:
        message += '<tr><td>' + instance + '</td>'
        for index in report[instance]:
            message += '<td>' + str(round(report[instance][index],2)) + '</td>'
        message += '</tr>'
    message += '</table><br>'
    message += '<div style="width:50%;float:left"><img src="img_loss" alt="Loss 曲线"></div>'
    message += '<div style="width:50%;float:left"><img src="img_accuracy" alt="accuracy 曲线"></div>'

    return HttpResponse(message)


def v_remove_train_file(request):
    remove_train_file(request.POST['filename'])
    return HttpResponseRedirect('model')


def v_remove_classify_file(request):
    remove_classify_file(request.POST['filename'])
    return HttpResponseRedirect('classifier')


def v_classify(request):
    report = classfy()
    message = '<table class="table table-striped"><tr>\n'
    message += '<th>报文类别</th>\n' + '<th>数量</th>\n'
    for k in report.items():
        message += '<tr><td>' + str(k[0]) + '</td>'
        message += '<td>' + str(k[1]) + '</td></tr>'
    message += '</table>'
    return HttpResponse(message)


def v_img_loss(request):
    with open('cnn/static/cnn/loss.png', 'rb') as f:
        img_loss= f.read()
    f.close()
    return HttpResponse(img_loss,content_type='image/png')


def v_img_accuracy(request):
    with open('cnn/static/cnn/accuracy.png', 'rb') as f:
        img_accuracy = f.read()
    f.close()
    return HttpResponse(img_accuracy, content_type='image/png')






