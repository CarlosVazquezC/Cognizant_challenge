# Cognizant_challenge

This dataset comprises several paper abstracts, one per file, that were furnished by the NSF
(National Science Foundation). A sample abstract is shown at the end.
Your task is developing an unsupervised model which classifies abstracts into a topic (discover
them!). Indeed, your goal is to group abstracts based on their semantic similarity.
You can get a sample of abstracts here. Be creative and state clearly your approach. Although
we don’t expect accurate results we want to identify your knowledge over traditional and newest
method over NLP

# Introduction

Topic modeling is one of several used NLP techniques which is frequently applied in document retrieval, personalizing content, trends identification.  Top modeling has four principal methods for topic extraction: LDA, NMF, Top2Vec and BERTopic.  

    Latent Dirichlet Allocation (LDA):  is an unsupervised technique for uncovering hidden topics within a document. By using Dirichlet distributions, LDA represents documents as distributions of topics and topics as distributions of words based on their frequency in the corpus. LDA assumes that each document contains a mixture of topics, and that each topic is a mixture of words that are associated with each other.

    Non-Negative Matrix Factorization (NMF):  is a linear algebra algorithm used to uncover hidden topics by decomposing high-dimensional vectors into non-negative, lower-dimensional representations. Given a normalized TF-IDF matrix as input, NMF outputs two matrices: a matrix of words by topics and a matrix of topics by documents.  Through multiple iterations, NMF optimizes the product of these two matrices until it reaches the original TF-IDF matrix.

    Top2Vec:  automatically detects high density areas of documents in its semantic space, the centroids of which are identified as the prominent topics in the corpora. Top2Vec assumes that each document is based on one topic, instead of a mixture of topics.

    BERTopic: Like Top2Vec, BERTopic uses BERT embeddings and a class-based TF-IDF matrix to discover dense clusters in the document corpora. These dense clusters allow for easily interpretable topics while keeping the most important words in the topic description.

