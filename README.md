# Natural Language Processing: from tokens to transformers

The notebooks of the CourseLabs course **Natural Language Processing**, ready to
run in Google Colab. Each lab is a folder; the shared data is in `data/`.

Click a badge to open a notebook in Colab, then run its first code cell. That
cell clones this repository, moves into the lab's folder, links `data/`,
installs the few packages Colab lacks and downloads the NLTK data, then prints
one line saying it is ready. Models download from Hugging Face as the notebook
first uses them. In a CourseLabs session the same cell does nothing, because
everything is already installed there.

To work on your own computer instead, download a lab's zip (the folder with its
data) or [all the data](https://github.com/fenago/nlp-course/releases/download/v1/data.zip), install the lab's
`requirements.txt`, and open the notebooks in Jupyter.

The checks in each notebook run anywhere. The one in Lab 11 that scores your
fine-tuned model against held-out answers runs only in CourseLabs, and so do
the checkpoint and the sign-off.

## The labs

### 01. Natural Language Processing 01: Why Can't a Computer Read?

Folder: [`lab-nlp-01-why-cant-a-computer-read/`](lab-nlp-01-why-cant-a-computer-read/). Download: [lab-nlp-01-why-cant-a-computer-read.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-01-why-cant-a-computer-read.zip) (the folder with its data).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-01-why-cant-a-computer-read/01_01_Tokens.ipynb) `01_01_Tokens.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-01-why-cant-a-computer-read/01_02_Normalising_Words.ipynb) `01_02_Normalising_Words.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-01-why-cant-a-computer-read/01_03_Tags_and_Entities.ipynb) `01_03_Tags_and_Entities.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-01-why-cant-a-computer-read/01_04_On_Your_Own.ipynb) `01_04_On_Your_Own.ipynb`

Credits: based on Chapter 1 of Dr. Ernesto Lee's NLP book; Kittiwake Mobile data, invented for the course; [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct); [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased).

### 02. Natural Language Processing 02: Turning Words into Numbers

Folder: [`lab-nlp-02-turning-words-into-numbers/`](lab-nlp-02-turning-words-into-numbers/). Download: [lab-nlp-02-turning-words-into-numbers.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-02-turning-words-into-numbers.zip) (the folder with its data).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-02-turning-words-into-numbers/02_01_Bag_of_Words.ipynb) `02_01_Bag_of_Words.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-02-turning-words-into-numbers/02_02_TF_IDF.ipynb) `02_02_TF_IDF.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-02-turning-words-into-numbers/02_03_Similarity_and_Search.ipynb) `02_03_Similarity_and_Search.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-02-turning-words-into-numbers/02_04_On_Your_Own.ipynb) `02_04_On_Your_Own.ipynb`

Credits: based on Chapter 2 of Dr. Ernesto Lee's NLP book; Kittiwake Mobile data, invented for the course; SMS Spam Collection, Almeida and Gomez Hidalgo, UCI, CC BY 4.0, https://archive.ics.uci.edu/dataset/228/sms+spam+collection; [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2).

### 03. Natural Language Processing 03: Teaching a Machine What Spam Looks Like

Folder: [`lab-nlp-03-teaching-a-machine-what-spam-looks-like/`](lab-nlp-03-teaching-a-machine-what-spam-looks-like/). Download: [lab-nlp-03-teaching-a-machine-what-spam-looks-like.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-03-teaching-a-machine-what-spam-looks-like.zip) (the folder with its data).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-03-teaching-a-machine-what-spam-looks-like/03_01_Spam_and_the_Baseline.ipynb) `03_01_Spam_and_the_Baseline.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-03-teaching-a-machine-what-spam-looks-like/03_02_Logistic_Regression.ipynb) `03_02_Logistic_Regression.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-03-teaching-a-machine-what-spam-looks-like/03_03_Support_Vector_Machines.ipynb) `03_03_Support_Vector_Machines.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-03-teaching-a-machine-what-spam-looks-like/03_04_On_Your_Own.ipynb) `03_04_On_Your_Own.ipynb`

Credits: based on Chapter 3 of Dr. Ernesto Lee's NLP book; Kittiwake Mobile data, invented for the course; SMS Spam Collection, Almeida and Gomez Hidalgo, UCI, CC BY 4.0, https://archive.ics.uci.edu/dataset/228/sms+spam+collection; [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2).

### 04. Natural Language Processing 04: Which Classifier, and Why?

Folder: [`lab-nlp-04-which-classifier-and-why/`](lab-nlp-04-which-classifier-and-why/). Download: [lab-nlp-04-which-classifier-and-why.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-04-which-classifier-and-why.zip) (the folder with its data).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-04-which-classifier-and-why/04_01_Nearest_Neighbours.ipynb) `04_01_Nearest_Neighbours.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-04-which-classifier-and-why/04_02_Naive_Bayes.ipynb) `04_02_Naive_Bayes.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-04-which-classifier-and-why/04_03_Choosing_a_Classifier.ipynb) `04_03_Choosing_a_Classifier.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-04-which-classifier-and-why/04_04_On_Your_Own.ipynb) `04_04_On_Your_Own.ipynb`

Credits: based on Chapter 4 of Dr. Ernesto Lee's NLP book; Kittiwake Mobile data, invented for the course; UCI Sentiment Labelled Sentences, Kotzias et al. (2015), CC BY 4.0, https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences; [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2).

### 05. Natural Language Processing 05: Words as Points in Space

Folder: [`lab-nlp-05-words-as-points-in-space/`](lab-nlp-05-words-as-points-in-space/). Download: [lab-nlp-05-words-as-points-in-space.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-05-words-as-points-in-space.zip) (the folder with its data).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-05-words-as-points-in-space/05_01_Pretrained_Word_Vectors.ipynb) `05_01_Pretrained_Word_Vectors.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-05-words-as-points-in-space/05_02_Skip_Gram_in_PyTorch.ipynb) `05_02_Skip_Gram_in_PyTorch.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-05-words-as-points-in-space/05_03_Products_as_Words.ipynb) `05_03_Products_as_Words.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-05-words-as-points-in-space/05_04_On_Your_Own.ipynb) `05_04_On_Your_Own.ipynb`

Credits: based on Chapter 5 of Dr. Ernesto Lee's NLP book; Kittiwake Mobile data, invented for the course; UCI Online Retail, Daqing Chen (2015), CC BY 4.0, https://archive.ics.uci.edu/dataset/352/online+retail; GloVe 6B 50d, Pennington, Socher and Manning (2014), PDDL 1.0, https://nlp.stanford.edu/projects/glove/.

### 06. Natural Language Processing 06: A Neuron from Scratch

Folder: [`lab-nlp-06-a-neuron-from-scratch/`](lab-nlp-06-a-neuron-from-scratch/). Download: [lab-nlp-06-a-neuron-from-scratch.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-06-a-neuron-from-scratch.zip) (the folder with its data).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-06-a-neuron-from-scratch/06_01_One_Neuron.ipynb) `06_01_One_Neuron.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-06-a-neuron-from-scratch/06_02_XOR_and_Hidden_Layers.ipynb) `06_02_XOR_and_Hidden_Layers.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-06-a-neuron-from-scratch/06_03_Activations_and_Loss.ipynb) `06_03_Activations_and_Loss.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-06-a-neuron-from-scratch/06_04_On_Your_Own.ipynb) `06_04_On_Your_Own.ipynb`

Credits: based on Chapter 6 of Dr. Ernesto Lee's NLP book; Kittiwake Mobile data, invented for the course; [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2).

### 07. Natural Language Processing 07: How a Network Learns

Folder: [`lab-nlp-07-how-a-network-learns/`](lab-nlp-07-how-a-network-learns/). Download: [lab-nlp-07-how-a-network-learns.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-07-how-a-network-learns.zip) (the folder with its data).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-07-how-a-network-learns/07_01_Backprop_by_Hand.ipynb) `07_01_Backprop_by_Hand.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-07-how-a-network-learns/07_02_Optimisers_Raced.ipynb) `07_02_Optimisers_Raced.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-07-how-a-network-learns/07_03_A_Recurrent_Network.ipynb) `07_03_A_Recurrent_Network.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-07-how-a-network-learns/07_04_On_Your_Own.ipynb) `07_04_On_Your_Own.ipynb`

Credits: based on Chapter 7 of Dr. Ernesto Lee's NLP book; Reuters-21578 ApteMod, downloaded by NLTK, CC BY 4.0, https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection.

### 08. Natural Language Processing 08: Remembering Across a Sentence

Folder: [`lab-nlp-08-remembering-across-a-sentence/`](lab-nlp-08-remembering-across-a-sentence/). Download: [lab-nlp-08-remembering-across-a-sentence.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-08-remembering-across-a-sentence.zip) (the folder with its data).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-08-remembering-across-a-sentence/08_01_Inside_an_LSTM_Cell.ipynb) `08_01_Inside_an_LSTM_Cell.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-08-remembering-across-a-sentence/08_02_LSTM_against_the_RNN.ipynb) `08_02_LSTM_against_the_RNN.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-08-remembering-across-a-sentence/08_03_Reading_Both_Ways.ipynb) `08_03_Reading_Both_Ways.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-08-remembering-across-a-sentence/08_04_On_Your_Own.ipynb) `08_04_On_Your_Own.ipynb`

Credits: based on Chapter 8 of Dr. Ernesto Lee's NLP book; UCI News Aggregator, Gasparetti (2016), CC BY 4.0, https://archive.ics.uci.edu/dataset/359/news+aggregator.

### 09. Natural Language Processing 09: From One Language to Another

Folder: [`lab-nlp-09-from-one-language-to-another/`](lab-nlp-09-from-one-language-to-another/). Download: [lab-nlp-09-from-one-language-to-another.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-09-from-one-language-to-another.zip) (the folder with its data).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-09-from-one-language-to-another/09_01_Encoder_and_Decoder.ipynb) `09_01_Encoder_and_Decoder.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-09-from-one-language-to-another/09_02_Greedy_and_Beam.ipynb) `09_02_Greedy_and_Beam.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-09-from-one-language-to-another/09_03_The_Bottleneck.ipynb) `09_03_The_Bottleneck.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-09-from-one-language-to-another/09_04_On_Your_Own.ipynb) `09_04_On_Your_Own.ipynb`

Credits: based on Chapter 9 of Dr. Ernesto Lee's NLP book; English-Spanish pairs from the Tatoeba Project, CC BY 2.0 FR, https://tatoeba.org, as packaged by https://www.manythings.org/anki/.

### 10. Natural Language Processing 10: Attention Is the Whole Trick

Folder: [`lab-nlp-10-attention-is-the-whole-trick/`](lab-nlp-10-attention-is-the-whole-trick/). Download: [lab-nlp-10-attention-is-the-whole-trick.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-10-attention-is-the-whole-trick.zip) (the folder with its data).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-10-attention-is-the-whole-trick/10_01_Attention_by_Hand.ipynb) `10_01_Attention_by_Hand.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-10-attention-is-the-whole-trick/10_02_A_Transformer_Block.ipynb) `10_02_A_Transformer_Block.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-10-attention-is-the-whole-trick/10_03_What_BERT_Looks_At.ipynb) `10_03_What_BERT_Looks_At.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-10-attention-is-the-whole-trick/10_04_On_Your_Own.ipynb) `10_04_On_Your_Own.ipynb`

Credits: based on Chapter 10 of Dr. Ernesto Lee's NLP book; UCI News Aggregator, Gasparetti (2016), CC BY 4.0, https://archive.ics.uci.edu/dataset/359/news+aggregator; [google/bert_uncased_L-4_H-256_A-4](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4).

### 11. Natural Language Processing 11: BERT and What Came After

Folder: [`lab-nlp-11-bert-and-what-came-after/`](lab-nlp-11-bert-and-what-came-after/). Download: [lab-nlp-11-bert-and-what-came-after.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-11-bert-and-what-came-after.zip) (the folder with its data).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-11-bert-and-what-came-after/11_01_Masked_Language_Modelling.ipynb) `11_01_Masked_Language_Modelling.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-11-bert-and-what-came-after/11_02_Fine_Tuning_BERT.ipynb) `11_02_Fine_Tuning_BERT.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-11-bert-and-what-came-after/11_03_Question_Answering.ipynb) `11_03_Question_Answering.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-11-bert-and-what-came-after/11_04_On_Your_Own.ipynb) `11_04_On_Your_Own.ipynb`

Credits: based on Chapter 11 of Dr. Ernesto Lee's NLP book; Kittiwake Mobile data, invented for the course; [distilbert/distilbert-base-uncased-distilled-squad](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad); [google/bert_uncased_L-2_H-128_A-2](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2); [google/bert_uncased_L-4_H-256_A-4](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4).

### 12. Natural Language Processing 12: Large and Small Language Models

Folder: [`lab-nlp-12-large-and-small-language-models/`](lab-nlp-12-large-and-small-language-models/). Download: [lab-nlp-12-large-and-small-language-models.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-12-large-and-small-language-models.zip) (the folder with its data).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-12-large-and-small-language-models/12_01_Next_Token.ipynb) `12_01_Next_Token.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-12-large-and-small-language-models/12_02_Instructions_and_Prompts.ipynb) `12_02_Instructions_and_Prompts.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-12-large-and-small-language-models/12_03_Retrieval_and_Grounding.ipynb) `12_03_Retrieval_and_Grounding.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fenago/nlp-course/blob/main/lab-nlp-12-large-and-small-language-models/12_04_On_Your_Own.ipynb) `12_04_On_Your_Own.ipynb`

Credits: written for CourseLabs; Kittiwake Mobile data, invented for the course; [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct); [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2).

### 13. Natural Language Processing 13: Models and a Database in the Browser

Folder: [`lab-nlp-13-models-and-a-database-in-the-browser/`](lab-nlp-13-models-and-a-database-in-the-browser/). Download: [lab-nlp-13-models-and-a-database-in-the-browser.zip](https://github.com/fenago/nlp-course/releases/download/v1/lab-nlp-13-models-and-a-database-in-the-browser.zip) (the folder with its data).

- A browser app: it does not run in Colab. Download the zip and follow its `README.md` to run it from a local static server.

Credits: written for CourseLabs.

## Licence

See [LICENSE.md](LICENSE.md): the course material is CourseLabs', and the data
keeps its original licences, with a link to each original.
