
#BLEU score

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu  #for sanity checks




def getcorpus_bleuscore(ref, hyp):
  cc = SmoothingFunction()
  return corpus_bleu(ref, hyp, smoothing_function=cc.method4)



def getsentence_bleuscore(ref, hyp):
    cc = SmoothingFunction()
    return sentence_bleu([ref], hyp, smoothing_function=cc.method2)
