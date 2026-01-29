from nltk import word_tokenize
from nltk.wsd import lesk
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))



def lesk_disambiguate(word, sentence, pos=None):
    """Disambiguazione con algoritmo di Lesk classico (wrapper semplice).

    Rimuove le stop words dal contesto prima di inviare i token a NLTK Lesk.
    Restituisce una tupla (synset_name|None, score)
    """
    try:
        tokens = word_tokenize(sentence)
        # Usa stop_words se definito, altrimenti fallback a set()
        filtered_tokens = [t for t in tokens if t not in stop_words]
        synset = lesk(filtered_tokens, word, pos=pos)
        return (synset.name() if synset else None, 0.0)
    except Exception as e:
        import traceback
        print("Lesk disambiguate exception occurred:", repr(e))
        traceback.print_exc()
        return (None, 0.0)