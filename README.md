# IA-semeval-2020-task12
Task 12: OffensEval 2: Identifying and Categorizing Offensive Language in Social Media (Twitter)

## Task 12
* Com o advento das principais plataformas de mídia social, preocupações crescentes cercam a segurança e a experiência on-line do usuário. Participo da tarefa compartilhada OffensEval do SemEval-2019, que usa o conjunto de dados de identificação de idioma ofensivo (OLID; Zampieri et al., 2019) para identificar linguagem ofensiva e abusiva nos tweets*. A seguir, siga o funcionamento das duas tasks:

          A) Identificação entre ofensiva(**OFF**- se contém palavrões ou uma ofensa direcionada, seja insultos, ameaças e conteúdos profanos) e não ofensivas(**NOT**) do idioma ;
	  B) O foco estava no TIPO do conteúdo ofensivo no post OU SEJA categorização automática de tipos de ofensas-- Insulto direcionado(**TIN** - targeted insult): Posts que contêm um insulto/ameaça a um indivíduo, grupo ou outros E o Não segmentado(**UNT** - untargeted): postagens que não contêm palavroẽs e xingamentos direcionados;
	  C) O sistema teve que detectar o alvo da mensagem nos posts ofensivos( se era destinada a um grupo ou a um indivíduo) OU SEJA identificação do alvo da ofensa-- usado apenas nas postagens que são TIN. Podem ser Individuais (**IND** - pessoa famosa, nome de um indivíduo ou participante sem nome no diálogo) , grupo (**GRP** - mesma etnia, gênero ou sexo, filiação política, religião crença ou outra característica) ou Outros(**OTH**): não pertence a nenhuma das anteriores, ex. uma organização, uma situação evento ou problema;


# Embeddings Pré-Treinados 
Você precisa fazer o download de word embedding pré-treinadas. Usamos o Twitter Glove, que pode ser encontrado aqui: https://nlp.stanford.edu/projects/glove/. Depois de fazer o download das incorporação, você pode executar o `` python model.py``

## Minha Abordagem
A abordagem envolveu dividir esse desafio em duas partes: processamento e amostragem de dados e escolha da arquitetura ideal de aprendizado profundo. Como nossos conjuntos de dados são dados de texto não estruturados e informais de mídias sociais, decidimos gastar mais tempo criando nosso pipeline de pré-processamento de texto para garantir que estamos alimentando dados de alta qualidade para o nosso modelo. **No meu modelo Deep learnig da Task A, utilizei 1 camada de embeddings, 4 camadas oucultas( 1 LSTM-RNN e 3 BiLSTM-RNN) e uma camada Densa.**

