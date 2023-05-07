Download Link: https://assignmentchef.com/product/solved-nlp-assignment1-nlp-techniques-to-improve-arabic-dialect-identification
<br>
The objective of this project is to apply NLP techniques in order to improve Arabic dialect identification on user generated content dataset.

In this assignment, you are provided with a large-scale collection of parallel sentences in the travel domain covering the dialects of 25 cities from the Arab World plus Modern Standard Arabic (MSA). The task is to build systems that predict a dialect class among one of the 26 labels (25+ MSA) for given sentences.

<h1>Dataset</h1>

The data of this assignment is the same reported on in the following papers.

Bouamor, H., Habash, N., Salameh, M., Zaghouani, W., Rambow, O., et al. (2018). The

MADAR Arabic Dialect Corpus and Lexicon. In Proceedings of the 11th International Conference on Language Resources and Evaluation. (PDF: <u>http://www.lrecconf.org/proceedings/lrec2018/pdf/351.pdf</u>)

Salameh, M., Bouamor, H. &amp; Habash, N. (2018). Fine-Grained Arabic Dialect Identification. In Proceedings of the 27th International Conference on Computational Linguistics. (PDF:

<u>http://aclweb.org/anthology/C18-1113</u>)

Systems’     Evaluation     Details

Evaluation will be done by calculating microaveraged F1 score (F1µ) for all dialect classes on the submissions made with predicted class of each sample in the Test set. To be precise, we define the scoring as following:




Pµ            =           ΣTPi           /            Σ(TPi           +           FPi)i           {Happy,Sad,Angry}

Rµ = ΣTPi / Σ(TPi + FNi)i {Happy,Sad,Angry}




where TPi is the number of samples of class i which are correctly predicted, FNi and FPi are the counts of Type-I and Type-II errors respectively for the samples of class i.




The final metric F1µ will be calculated as the harmonic mean of Pµ and Rµ.




<h1>Implementation</h1>




<h2>Step1 : Data preprocessing</h2>




Download the Training and Development Data Canvas. And pre-process the dataset, i.e. cleaning, tokenization, etc.

You can find an Arabic text tokenizer (Farasa) within the Project_1 folder.

You can, also, use Camel Tools (already installed in C127 machines).

<u>https://github.com/CAMeL-Lab/camel_tools</u>




<h2>Step2 : System implementation</h2>




Your task is to implement and compare three different text classification methods as follows.




<strong>1-</strong><strong> Feature-Based Classification for Dialectal Arabic (20%) </strong>

<strong> </strong>

Use and compare <strong>two different feature-based</strong> classification methods (classical Machine Learning techniques) in order to implement your Arabic dialect identification system. Your models should apply various n-gram features as follows:

<ul>

 <li>Word-gram features with uni-gram, bi-gram and tri-gram;</li>

 <li>Character-gram features with/without word boundary consideration, from bi-gram and up to 5-gram.</li>

</ul>




<strong>2-</strong><strong> LSTM Deep Network </strong>

<strong> </strong>

Use the Long Short Term Memory  (LSTM) architecture with <strong>AraVec</strong> pre-trained word embeddings models.  These models were built using <u>gensim</u> Python library. Here’s  the steps for using the tool:

<ol>

 <li>Install gensim &gt;= <strong>4</strong> and nltk &gt;= <strong>3.2</strong> using either pip or conda</li>

</ol>

pip install gensim nltk conda install gensim nltk

<ol start="2">

 <li>extract the compressed model files to a directory [ e.g. Twittert-CBOW ]</li>

 <li>keep the <strong>.npy</strong> You are gonna to load the file with no extension, like what you’ll see in the following figure.</li>

 <li>run the python code to load and use the model</li>

</ol>







You can find a simple code for loading and using one the models by following these steps in the following link:

<u>https://github.com/bakrianoo/aravec</u>

And an example of using LSTM for text classification in the following guide:

<u>https://towardsdatascience.com/multi-class-text-classification-with-lstm1590bee1bd17</u>







<strong>3-</strong><strong> BERT for Dialectal Arabic </strong>

<strong> </strong>

BERT<sup>1</sup> or Bidirectional Encoder Representations from Transformers (BERT), has recently been introduced by Google AI Language researchers (Devlin et al., 2018). It replaces the sequential nature of RNN (LSTM &amp; GRU) with a much faster Attention-based approach. The model is also pre-trained on two unsupervised tasks, masked language modeling and next sentence prediction. This allows you to use a pre-trained BERT model by fine-tuning the same on downstream specific tasks such as Dialectal Arabic classification.




You can employ the multi-lingual BERT that has been pre-trained on MSA and then fine tune it for dialectal Arabic.

I recommend to read the following Blog Multi-label Text Classification using BERT <u>https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mightytransformer-69714fa3fb3d</u>







<strong>4-</strong>  <strong>Evaluation (10%) </strong>

Use the MADAR-DID-Scorer.py scrip to evaluate your systems.







<h2>Steaming and Classifying Arabic Tweets from Twitter (Bonus)</h2>

For those of you that might be interested in applying the Dialectal Arabic classifier in real classification of Arabic tweets, it is possible to create a developer account with Twitter.

You can use the Twitter API to stream tweets based on a specific keyword.

Go to the following for a <u>guide</u>. <u>https://pythonprogramming.net/twitter-api-streaming-tweets-python-tutorial/</u>

Please note it can take a little bit of time to get this working. You can pipe these tweets through your model in order to classify sentiment. For example, you could increase the probability threshold for both positive and negative and classify sporting events and monitor/graph sentiment throughout the event.




<strong><u>Reference:</u></strong>

Jacob               Devlin,   Ming-Wei              Chang,    Kenton   Lee,         and         Kristina Toutanova.           2018.         Bert:       Pre-training          of            deep       bidirec-  tional     transformers        for           language         understanding.     <em>arXiv      preprint arXiv:1810.04805</em>.

<strong>      </strong>

<span style="text-decoration: line-through;">                                              </span>

<sup>1</sup> For                more       technical details     see          (Devlin   et             al.,           2018).

Source:            https://mc.ai/bert-explained-state-of-the-art-language-model-for-nlp/

Source:            http://jalammar.github.io/illustrated-bert/

BERT                multi-lingual          mdoel     is             presented              in             https://github.com/google-research/bert/blob/master/multilingual.md