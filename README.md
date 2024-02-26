# Cognizant_challenge

This dataset comprises several paper abstracts, one per file, that were furnished by the NSF
(National Science Foundation). A sample abstract is shown at the end.
Your task is developing an unsupervised model which classifies abstracts into a topic (discover
them!). Indeed, your goal is to group abstracts based on their semantic similarity.
You can get a sample of abstracts here. Be creative and state clearly your approach. Although
we don’t expect accurate results we want to identify your knowledge over traditional and newest
method over NLP.


# 0- Introduction

Topic modeling is one of several used NLP techniques which is frequently applied in document retrieval, personalizing content, trends identification.  Top modeling has four principal methods for topic extraction: LDA, NMF, Top2Vec and BERTopic.  

    Latent Dirichlet Allocation (LDA):  is an unsupervised technique for uncovering hidden topics within a document. By using Dirichlet distributions, LDA represents documents as distributions of topics and topics as distributions of words based on their frequency in the corpus. LDA assumes that each document contains a mixture of topics, and that each topic is a mixture of words that are associated with each other.

    Non-Negative Matrix Factorization (NMF):  is a linear algebra algorithm used to uncover hidden topics by decomposing high-dimensional vectors into non-negative, lower-dimensional representations. Given a normalized TF-IDF matrix as input, NMF outputs two matrices: a matrix of words by topics and a matrix of topics by documents.  Through multiple iterations, NMF optimizes the product of these two matrices until it reaches the original TF-IDF matrix.

    Top2Vec:  automatically detects high density areas of documents in its semantic space, the centroids of which are identified as the prominent topics in the corpora. Top2Vec assumes that each document is based on one topic, instead of a mixture of topics.

    BERTopic: Like Top2Vec, BERTopic uses BERT embeddings and a class-based TF-IDF matrix to discover dense clusters in the document corpora. These dense clusters allow for easily interpretable topics while keeping the most important words in the topic description.

For the solution of this challenge I opted to use a modification of the BERTopic for a better validation of the steps and results.  In the next paragraphs each of the steps required for the topic extraction will be described.


# 1- First steps in code

In this part the libraries required for the different steps are implemented: extraction of the abstracts from the xml, transformation of sentences to vectors, comparissons among vectors, etc.


#  2- Abstracts cleanning and formatting

Even when methods as BERTopic do not required preprocessing (stop words removal, tokenization, stemming, lemmatization, etc), it is necessary to extract all the xml remnants in the paragraphs.  Also, embedding process requires that abstracts must be presented as a list.  Next cell cleans the abstracts, remove the ones without narration and prepares the list.  Once that abstracts were processed in this small procedure, now it follows the embedding procedure.


# 3. Pretrained model selection

Sentence_transformers framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images. The models are based on transformer networks like BERT / RoBERTa / XLM-RoBERTa etc. and achieve state-of-the-art performance in various tasks. Text is embedded in vector space such that similar text are closer and can efficiently be found using cosine similarity.
For this experiment I use three different pretrained models.  At the end, all-MiniLM-L12-v2 resulted to have a great embedding performance, with a reasonable speed and compact size.  For a better reference of the pretrained models please verify the link https://www.sbert.net/docs/pretrained_models.html
Sentence_transformers provide a large list of Pretrained Models for more than 100 languages. Some models are general purpose models, while others produce embeddings for specific use cases. Pre-trained models can be loaded by just passing the model name: SentenceTransformer('model_name').


# 4. Embedding procedure

The next step is converting the documents to numerical data. I used BERT for this purpose as it extracts different embeddings based on the context of the word.  After this step, each of the abstracts are transformed to 384-dimensional vectors.  It is necessary to validate that similar topics are clustered together such that we can find the topics within these clusters. Before that, we need to lower the dimensionality of the vectors.


# 5. UMAP (Uniform Manifold Approximation and Projection)

UMAP package help us to reduce the dimensionality while keeping the size of the local neighborhood


# 6. HDBSAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)

HDBSCAN learns the context of the language and identifies high density regions within a document corpora based on their word distributions to deduce the optimal number of topics.  At the end of this step, result dataframe shows the label of each abstract (cluster number) and their respective coordinates.


# 7. Topic creation

If it is used a variant of TF-IDF , that would allow me to extract what makes each set of documents unique compared to the other.  To create this class-based TF-IDF score, we need to first create a single document for each cluster of documents.  The frequency of each word is extracted for each class and divided by the total number of words. This action can be seen as a form of regularization of frequent words in the class. Next, the total, unjoined, number of documents is divided by the total frequency of words across all classes.  With this function now we have a single importance value for each word in a cluster which can be used to create the topic. If we take the top 5 most important words in each cluster, then we would get a good representation of a cluster, and thereby a topic.


# 8. Topic representation

In order to create a topic representation, we take the top 20 words per topic based on their c-TF-IDF scores. The higher the score, the more representative it should be of its topic as the score is a proxy of information density.  The topic name-1 refers to all documents that did not have any topics assigned. The great thing about HDBSCAN is that not all documents are forced towards a certain cluster. If no cluster could be found, then it is simply an outlier.  This is an example of the cluster 19 and the top 10 words:

    [('ms', 0.04267711119828454),
     ('mass', 0.037245684475675785),
     ('spectrometer', 0.031208950038714642),
     ('ions', 0.021001849288100116),
     ('spectrometry', 0.020786882877533375),
     ('lc', 0.017810556219059085),
     ('ion', 0.016217981590087152),
     ('instrument', 0.015897025052069576),
     ('instrumentation', 0.014692004753945841),
     ('analytical', 0.01430578446885847)]


# 9. Topic reduction

One practical step is the reduction of topics, which consist in the reduction of the topics number by merging the topic vectors that were most similar to each other.  We can use a similar technique by comparing the c-TF-IDF vectors among topics, merge the most similar ones, and finally re-calculate the c-TF-IDF vectors to update the representation of our topics.  The final dataframe contains the name of the topic, its size and the cluster name consisting in the concatenation of the first five topics.


    Topic	Size	Cluster_name
    -1	-1	4115.0	ABSTRACTS-WITHOUT-CLUSTER
    479	478.0	133.0	covid - pandemic - 19 - risk - social
    65	64.0	122.0	archaeological - heritage - indigenous - ancie...
    84	83.0	108.0	income - graduation - scholarships - scholars ...
    196	195.0	102.0	memory - cloud - hardware - performance - serv...
    ...	...	...	...
    238	237.0	6.0	br - optimizers - ml - machine - hydrology
    92	91.0	6.0	writing - tutors - tutor - assignment - waes
    300	299.0	6.0	inclusive - practices - classrooms - faculty -...
    90	89.0	6.0	mantle - deformation - visco - motions - plates
    355	354.0	6.0	launch - mold - propulsion - satellite - leo


# 10. Conclusions

The aim of this code was to develope an unsupervised model which classifies abstracts into a topic based on their semantic similarity.  There are some comments about the presented code:

    The use of the sentence-transformer package was crucial to this task.  It generated high-quality embeddings.
    Pre-trained model was selected according to the speed and performance; however there were not exhaustive tests to assure that it is not the best of the options.
    HDBSAN is a great option for the detection of topics because it does not force data points to clusters as it considers them outliers.
    I decided to use just the principal 5 words for the topic definition, but it could use more words and it would offer a better description of the cluster.
    Topic reduction could be not strictly necessary.  I considered that due to the redundancy in some of the clusters it would be a good idea to maintain a low count of clusters.


# 11. References

    Deerwester, S. (1990). Indexing by latent semantic analysis. Retrieved from Asis&t: https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9

    Hofmann, T. (1999). Probabilistic latent semantic analysis. Retrieved from ACM: https://dl.acm.org/doi/10.5555/2073796.2073829

    Blei, D. (2003). Latent Dirichlet Allocation. Retrieved from Journal of ML Research: https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

    Angelov, D. (2020). Top2Vec: Distributed Representations of Topics. Arxiv.

    Briggs, J. (n.d.). Advanced Topic Modeling with BERTopic. Retrieved from Pinecone: https://www.pinecone.io/learn/bertopic/

    David, D. (2021, August 24). NLP Tutorial: Topic Modeling in Python with BerTopic. Retrieved from hackernoon: https://hackernoon.com/
    nlp-tutorial-topic-modeling-in-python-with-bertopic-372w35l9

    Grootendorst, M. (2020, October 5). Topic Modeling with BERT.  Leveraging BERT and TF-IDF to create easily interpretable topics. Retrieved from Towards Data Science: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

    Grootendorst, M. (2021, January 6). Interactive Topic Modeling with BERTopic.  An ind-depth guide to topic modeling with BERTopic. Retrieved from Towards Data Science: https://towardsdatascience.com/interactive-topic-modeling-with-bertopic-1ea55e7d73d8














