#@AUTHOR: Cleiton Solano

#PREPROCESSAMENTO!!!
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import re

#RESULTADOS!!!
from sklearn.metrics import classification_report

#MATRIZ DOS WORD_EMBEDDINGS
import numpy
from numpy import array, asarray, zeros, split
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import  train_test_split
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout, Bidirectional, GRU, SpatialDropout1D
import sklearn.metrics as metrics

TEST_SIZE = None
DATA_SIZE = None
global vocab_length_teste
global vocab_length_treino
global maior1
global maior2


class Preprocessamento(object):
    def __init__(self):
        self.all_tweets = None
        self.polarity_tweets = None

    def read_tweets_from_file(self, dataset):
        self.all_tweets = dataset['tweet']

        return self.all_tweets

    def read_polarity_from_file(self, dataset):
        self.polarity_tweets = dataset['subtask_a']

        return self.polarity_tweets

    def tokenizando(self, tweet):

        tweet = tweet.lower()
        #word_tokenizer = RegexpTokenizer("[\w']+")  # elimina mantem can't junto e elimina pontuação
        token = RegexpTokenizer('\s+',gaps = True)
        tweet = token.tokenize(tweet)

        return tweet#(" ".join(tweet))

    def tokenizando_embeddings(self, corpus, tipo):

        word_tokenizer = Tokenizer()
        word_tokenizer.fit_on_texts(corpus)

        # parâmetro do tamanho da sentença de entrada
        global vocab_length_teste
        global vocab_length_treino

        if tipo == 'treino':
            vocab_length_treino = len(word_tokenizer.word_index) + 1
        else:
            vocab_length_teste = len(word_tokenizer.word_index) + 1

        embedded_sentences = word_tokenizer.texts_to_sequences(corpus)
        #print(embedded_sentences)

        return embedded_sentences

    #pos_emo = [':)' , '(:' , ';)' , ':-)' , '(-:' ,':D ',':-D' , ':P' , ':-P']#POSITIVO
    #neg_emo =  [':(' , '):', ';(', ':-(', ')-:' , 'D:' , 'D-:' , ':’( ', ':’-( ', ')’: ', ')-’:']#NEGATIVO
    def limpando_users_links(self, tweet):

        flag = 0
        if ';(' or ';)' in tweet:
            flag = 1
        noises = ['URL', '@USER', '.', '?', ',', '!', '&', ';','"']
        for noise in noises:
            if noise == ';' and flag == 1:
                continue
            else:
                tweet = tweet.replace(noise, '')

        tweet = tweet.lower()

        return tweet
        #return re.sub(r'[^a-zA-Z]', ' ', tweet)# apenas deixando palavras !!! removendo qual quer outro tipo de simbulo....ACHO QUE VAI USAR
        # palavra f**king

    def remove_number(self, tweet):
        newTweet = re.sub('\\d+', '', tweet)
        return newTweet

    def remove_links(selfself, tweet):
        newTweet = re.sub(r'(https|http)?://(\w|\.|/|\?|=|&|%)*\b', '', tweet, flags=re.MULTILINE)
        return newTweet

    def remove_stopwords(self, tweet):

        #'do', 'my', 'wouldn', 'our', 'there', 'than', 're'....
        english_stops = set(stopwords.words('english'))
        token_limpo = [w for w in tweet if not w in english_stops]

        return token_limpo

    def lemma(self, tweet):

        lematização = WordNetLemmatizer()
        token_limpo = list()

        for token in tweet:
            token = lematização.lemmatize(token)
            if len(token) > 1:
                token_limpo.append(token) #https://github.com/alisonrib17/ACM-Paper/tree/master/English/Task

        return (" ".join(token_limpo))


class load_embeddings(object):
    def __init__(self):
        self.matrix = None

# ficar atento que essa metodologia recebe o corpus TOKENIZADO
    #def create_matrix(self, maior1, corpus_train, X_train, X_test, y_train, y_test):
    def create_matrix(self, maior1, corpus_train, X_train, y_train):
        embeddings_dicionario = dict()
# --------------------------------------------------------------------------------------------------*
        # usando 100 DIMENSõES https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/50350
        glove_file = open('/home/cleiton/PycharmProjects/glove.twitter.27B/glove.twitter.27B.100d.txt')

        for i in glove_file:
            records = i.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dicionario[word] = vector_dimensions

        glove_file.close()

        embedding_matrix1 = zeros((vocab_length_treino, 100))
        word_tok = Tokenizer()
        word_tok.fit_on_texts(corpus_train)


        for word, index in word_tok.word_index.items():
            embedding_vector = embeddings_dicionario.get(word)
            if embedding_vector is not None:
                embedding_matrix1[index] = embedding_vector

# --------------------------------------------------------------------------------------------------*

        ''' 
        https://www.kaggle.com/terenceliu4444/glove6b100dtxt#glove.6B.100d.txt
        http://nlp.stanford.edu/data/glove.6B.zip
        '''
        model = Sequential()
        embedding_layer = Embedding(vocab_length_treino, 100, weights=[embedding_matrix1], input_length=maior1, trainable=False)
        model.add(embedding_layer)

        '''Primeiro Teste'''
        '''model.add(LSTM(100, return_sequences=True, input_shape=(100,)))
        model.add(LSTM(100, return_sequences=True))
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(LSTM(100))'''
        #model.add(LSTM(150, dropout=0.25, recurrent_dropout=0.35))

        '''Segundo teste'''
        model.add(LSTM(200, return_sequences=True, input_shape=(180,), activation='tanh'))
        model.add(Bidirectional(LSTM(180, return_sequences=True, dropout=0.15, recurrent_dropout=0.35, activation='tanh')))#pesquiar sobre a relu
        model.add(Bidirectional(LSTM(180, return_sequences=True, dropout=0.15, recurrent_dropout=0.35, activation='tanh')))#pesquiar sobre a relu
        model.add(Bidirectional(LSTM(180, dropout=0.15, recurrent_dropout=0.35, activation='tanh')))

        '''LSTM para séries temporais , você deve ter Denso (1)'''
        model.add(Dense(1, activation='sigmoid'))
# ---------------------------------------------------------------------------------------------------------------------------------------------------

        seed = 7
        numpy.random.seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.27, shuffle=True, random_state=seed, stratify=y_train)

# ---------------------------------------------------------------------------------------------------------------------------------------------------

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) # , le.f1_m, le.precision_m, le.recall_m
        #print(model.summary())

        model.fit(X_train, y_train, validation_split=0.25, epochs=50, callbacks=None, batch_size=64)#, class_weight=[embedding_matrix1]
        loss, accuracy = model.evaluate(X_train, y_train, verbose=0)#(padded_sentences, sentiments)


        #y_pred1 = model.predict(X_test, batch_size=64, verbose=1)
        #y_pred = numpy.argmax(y_pred1, axis=1)

        print(f'Acuracia = {accuracy * 100}')
        print(f'Loss: {loss* 100}')

        y_pred = (model.predict(X_test, batch_size=64)>.5).astype(int)#, verbose=1
        #y_pred_bool = numpy.argmax(y_pred, axis=1) # CASO QUEIRA IMPRIMIR A TABELA DE MÉTRICAS
        
        #print(classification_report(y_test, y_pred))
        print('\n\n')
        print('--'*10, 'MÉTRICAS', '--'*10)
        print(f'* Precisão: {metrics.precision_score(y_test, y_pred, average="macro") * 100}')
        print(f'* Recall: {metrics.recall_score(y_test, y_pred, average="macro") * 100}')
        print(f'* F1: {metrics.f1_score(y_test, y_pred, average="macro") * 100}')
        print(f'* Acurácia: {metrics.accuracy_score(y_test, y_pred) * 100}')

        return self.matrix


class Gram(object):
    def __init__(self):
        self.bigram = None

    def create_bigram(self, tweet):

        self.bigram = []

        for i in range(len(tweet) - 1):
            b_gram = tweet[i] + "_" + tweet[i + 1]
            self.bigram.append(b_gram)

        return (" ".join(self.bigram))


class retira_polaridade(object):
    def __init__(self):
        self.all_tweets = None

    def retira_polaridade(self, corpus, task):
        self.all_tweets = []
        sentiment = list()
        if task == 'a':
            corpus.replace(to_replace=['OFF', 'NOT'], value=[0,1], inplace=True)
            for i in corpus:
                sentiment.append(int(i))
            sentiment = array(sentiment)
        elif task == 'b':
            corpus.replace(to_replace=['TIN', 'UNT'], value=[0, 1], inplace=True)
            for i in corpus:
                sentiment.append(int(i))
            sentiment = array(sentiment)
        elif task == 'c':
            corpus.replace(to_replace=['IND', 'GRP', 'OTH'], value=[0, 1, 2], inplace=True)
            for i in corpus:
                sentiment.append(int(i))
            sentiment = array(sentiment)

        self.all_tweets = sentiment

        return self.all_tweets


def main():

    global TEST_SIZE
    global DATA_SIZE

    OLID_train = pd.read_csv('/home/cleiton/PycharmProjects/OLIDv1.0/olid-training-v1.0.tsv', sep='\t', header=0, encoding='utf-8')
    OLID_test = pd.read_table('/home/cleiton/PycharmProjects/dados_novos/trial-data/offenseval-trial.txt', sep="\t", header=None, names=["tweet", "subtask_a", "subtask_b", "subtask_c"], encoding='utf-8')

    rp = retira_polaridade()
    pre = Preprocessamento()
    gram = Gram()
    load = load_embeddings()

    '''lendo a base de dados apenas a coluna do tweet'''
    tweets_train = pre.read_tweets_from_file(OLID_train)
    tweets_test = pre.read_tweets_from_file(OLID_test)

    '''lendo a base de dados apenas a coluna da task_a'''
    sentimentos1 = pre.read_polarity_from_file(OLID_train)
    sentimentos2 = pre.read_polarity_from_file(OLID_test)

    task = 'a'
    sentiment1 = rp.retira_polaridade(sentimentos1, task)
    sentiment2 = rp.retira_polaridade(sentimentos2, task)

    #testando se esta recebendo corretamente
    print(tweets_train[9])
    print(tweets_test[3])
    print(sentiment2[4])

    all_tweets = list()
    test_all_tweets = list()

    for tweet in tweets_train:
        all_tweets.append(tweet)

    for tweet in tweets_test:
        test_all_tweets.append(tweet)

    DATA_SIZE = len(all_tweets)
    TEST_SIZE = len(test_all_tweets)

    print(f'O tamanho total tweet"s de treino: {DATA_SIZE}')
    print(f'O tamanho total  tweet"s de teste: {TEST_SIZE}')
# -------------------------------------------------------------------------------
    # PREPROCESSAMENTO!!!!

    for i in range(len(all_tweets)):
        all_tweets[i] = pre.limpando_users_links(all_tweets[i])
        all_tweets[i] = pre.remove_number(all_tweets[i])
        all_tweets[i] = pre.remove_links(all_tweets[i])
        #all_tweets[i] = pre.tokenizando(all_tweets[i])
        #all_tweets[i] = pre.remove_stopwords(all_tweets[i]) # AQUI ESTÁ COMENTADO, POIS ESTOU SEGUINDO O MESMO PRÉ-PRCOCESSAMENTO DA CRIAÇÃO DO GLOVE
        #all_tweets[i] = pre.lemma(all_tweets[i])

    for i in range(len(test_all_tweets)):
        test_all_tweets[i] = pre.limpando_users_links(test_all_tweets[i])
        test_all_tweets[i] = pre.remove_number(test_all_tweets[i])
        test_all_tweets[i] = pre.remove_links(test_all_tweets[i])
        #test_all_tweets[i] = pre.tokenizando(test_all_tweets[i])
        #test_all_tweets[i] = pre.remove_stopwords(test_all_tweets[i]) # AQUI ESTÁ COMENTADO, POIS ESTOU SEGUINDO O MESMO PRÉ-PRCOCESSAMENTO DA CRIAÇÃO DO GLOVE
        #est_all_tweets[i] = pre.lemma(test_all_tweets[i])



    print('-----'*15, '\n TREINO: \n')
    print(all_tweets[2])
    print(all_tweets[3])
    print(all_tweets[9])
    print('-----' * 15, '\nTESTE:\n')
    print(test_all_tweets[2])
    print(test_all_tweets[3])
    print(test_all_tweets[4])

# -------------------------------------------------------------------------------*

    # CRIANDO OS N-GRAM
    print('----'*15)


    tweets_unigram1 = list()

    tweets_unigram1 = all_tweets
    tweets_unigram2 = test_all_tweets

    tweets_bigram1 = list()
    tweets_bigram2 = list()

    for i in range(len(all_tweets)):
        tweets_bigram1.append(gram.create_bigram(all_tweets[i].split()))

    for i in range(len(test_all_tweets)):
        tweets_bigram2.append(gram.create_bigram(test_all_tweets[i].split()))


    print(tweets_bigram1[3])
    print(tweets_bigram1[9])
    print(tweets_bigram2[5])
    print('----' * 15)

# -------------------------------------------------------------------------------*

    # CRIANDO A MATRIZ DE EMBEDDINGS

    '''TREINO'''
    sentences1 = all_tweets
    word_count = lambda sentence: len(word_tokenize(sentence))
    longest_sentence1 = max(sentences1, key=word_count)
    maior1 = len(word_tokenize(longest_sentence1))

    '''TESTE'''
    sentences2 = test_all_tweets
    word_count = lambda sentence: len(word_tokenize(sentence))
    longest_sentence2 = max(sentences2, key=word_count)
    maior2 = len(word_tokenize(longest_sentence2))
    

    print(f'Maior treinamento: {maior1}')
    print(f'Maior teste: {maior2}')

    all1 = pre.tokenizando_embeddings(all_tweets, 'treino')
    all2 = pre.tokenizando_embeddings(test_all_tweets, 'teste')

    print(f'Vocabulario treinamento: {vocab_length_treino}') # ELE DEU 19840
    print(f'Vocabulario test: {vocab_length_teste}') # ELE DEU 19840
    '''
    -Num de palavras únicas no CORPUS - me da o vocabulário = 19840.
    -Dimensão vetorial do vetor de saída = 100.
    -A camada de embeddings terá um número de parâmetros: 19840 * 100 = 1984000.
    -A camada de saída de embeddings será um vetor 2D com 57 linhas(1 para
    cada palavra na frase) e 100 colunas.
    - A saída da camada de embeddings será achatada para que possa ser usada 
    com a camada densa.
    '''

    print('----' * 15)
    #print(all_tweets)
    # deixei todos os vetores de palavras com o tamanho do maior vetor e preenchi com zeros
    padded_sentences1 = pad_sequences(all1, maior1, padding='post')
    padded_sentences2 = pad_sequences(all2, maior1, padding='post')
    print(padded_sentences2.shape)
    print(padded_sentences1.shape)

    '''
    x = load.create_matrix(maior1, all_tweets, padded_sentences1, padded_sentences2, sentiment1, sentiment2)
    def create_matrix(self, maior1, corpus_train, X_train, X_test, y_train, y_test):
    '''
    x = load.create_matrix(maior1, all_tweets, padded_sentences1, sentiment1)

    print('----' * 15)

# -------------------------------------------------------------------------------*

main()
