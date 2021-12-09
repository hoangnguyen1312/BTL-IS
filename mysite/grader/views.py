from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse

from .models import Question, Essay
from .forms import AnswerForm

from .utils.model import *
from .utils.helpers import *
from .utils.lda import *

import os
current_path = os.path.abspath(os.path.dirname(__file__))

# Create your views here.
def index(request):
    questions_list = Question.objects.order_by('set')
    context = {
        'questions_list': questions_list,
    }
    return render(request, 'grader/index.html', context)

def essay(request, question_id, essay_id):
    essay = get_object_or_404(Essay, pk=essay_id)
    context = {
        "essay": essay,
    }
    return render(request, 'grader/essay.html', context)

def question(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = AnswerForm(request.POST)
        if form.is_valid():

            content = form.cleaned_data.get('answer')

            # if len(content) > 20:
            print("Question id: ", question_id)
            idx_topic = check_topic(content)

            ref = {
                "1": "4",
                "2": "0",
                "3": "6",
                "4": "1",
                "5": "7",
                "6": "3",
                "7": "2",
                "8": "5"
            }

            num_features = 300
            model = word2vec.KeyedVectors.load_word2vec_format(os.path.join(current_path, "deep_learning_files/word2vec.bin"), binary=True)
            clean_test_essays = []
            clean_test_essays.append(essay_to_wordlist(content, remove_stopwords=True ))
            testDataVecs = getAvgFeatureVecs( clean_test_essays, model, num_features )
            testDataVecs = np.array(testDataVecs)
            testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

            lstm_model = get_model()
            lstm_model.load_weights(os.path.join(current_path, "deep_learning_files/final_lstm.h5"))
            preds = lstm_model.predict(testDataVecs)

            if math.isnan(preds):
                preds = 0
            else:
                preds = np.around(preds)

            if preds < 0:
                preds = 0
            elif preds > question.max_score:
                preds = question.max_score

            K.clear_session()
            if ref[str(question_id)] != str(idx_topic):
                final_score = 0
                notification = "Your post does not match this topic"
            else:
                score = question.max_score
                final_score = (score + preds) / 2
                notification = "Your post matches this topic"
            
            content = notification 
            essay = Essay.objects.create(
                content=content,
                question=question,
                score=final_score
            )
        return redirect('essay', question_id=question.set, essay_id=essay.id)
    else:
        form = AnswerForm()

    context = {
        "question": question,
        "form": form,
    }
    return render(request, 'grader/question.html', context)