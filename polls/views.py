from django.shortcuts import render,get_object_or_404

#导入django的http响应模块
from django.http import HttpResponse,HttpResponseRedirect
from .models import Question,Choice
from django.urls import reverse
from django.views import generic

# Create your views here.

def index(request):
    latest_question_list = Question.objects.order_by('pub_date')[:5]
    context = {'latest_question_list': latest_question_list}
    return render(request, 'polls/index.html', context)

def detail(request, question_id):
    question = get_object_or_404(Question,pk=question_id)
    return render(request,'polls/detail.html',{'question': question})

def result(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    return render(request, 'polls/result.html', {'question': question})
#投票函数
def vote(request, question_id):
    #根据页面请求中的question_id获得一个question，如果是404错误，则返回错误信息
    question = get_object_or_404(Question, pk=question_id)
    try:
        #用户的选择关联该问题，赋值给selected_choice
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    #如果选项不存在则抛出异常
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        #返回出错页面
        return render(request, 'polls/detail.html', {'question': question,'error_message': "You didn't select a choice.",})
    else:
        #该选项选票加1并保存进数据库
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        #完成操作后，重定向到一个详细页面
        return HttpResponseRedirect(reverse('polls:result', args=(question.id,)))

class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        return Question.objects.order_by('-pub_date')[:5]

class DetailView(generic.DetailView):
    model = Question
    template_name = 'polls/detail.html'

class ResultView(generic.DetailView):
    model = Question
    template_name = 'polls/result.html'



