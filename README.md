# Gen AI Project : Generative Search System (Mr.HelpMate AI)

## Table of Contents:
* [Introduction](#introduction)
* [Objective](#objective)
* [Dataset](#dataset)
* [System Architecture](#system-architecture) 
* [Testing and Evaluation](#testing-and-evaluation) 
* [Challenges Faced](#challenges-faced) 
* [Conclusion](#conclusion) 
* [Technologies Used](#technologies-used)
* [References](#references)
* [Contact Information](#contact-information)

## Introduction:  
<div align="justify"> Traditional search methods often struggle with efficiently extracting relevant information from complex and extensive documents, also leads to time-consuming, inaccurate, and unreliable results. To address these issues, we will develop a comprehensive and robust AI-based generative search system which will be capable of effectively and accurately answering questions from a PDF document. This innovative approach allows users to pose precise, context-aware questions and receive accurate answers directly from the text. By improving efficiency, accuracy, and accessibility, the system enhances document management and data retrieval, which may be useful in various sectors such as legal, financial, medical, and academic etc.</div> <br>

<div align="justify">  This system will utilize the Retrieval Augmented Generation (RAG) pipeline, which combines embedding, search and ranking, and generative layers to provide comprehensive and contextually relevant answers. </div>

### Retrieval-Augmented Generation (RAG):
<div align="justify"> RAG is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. Large Language Models (LLMs) are trained on vast volumes of data and use billions of parameters to generate original output for tasks like answering questions, translating languages, and completing sentences. It is a cost-effective approach to improving LLM output, so it remains relevant, accurate, and useful in various contexts.</div>

## Objective:  
<div align="justify"> The main objective of this project is to develop a robust generative search system which will be capable of effectively and accurately answering questions from a document. i.e. PDF document.</div>

## Dataset:  
#### Document:
- The project will use a long life insurance policy document titled "Principal-Sample-Life-Insurance-Policy"

#### File Format:
- The document is provided in PDF format.

## System Architecture:
The project is based on the Retrieval Augmented Generation (RAG) pipeline which is structured into three main layers:

-  Embedding Layer
-  Search Layer
-  Generation Layer

Each layer plays a critical role to ensure the system's ability to accurately retrieve and generate responses from the pdf document.
  
### 1. Embedding Layer:

<div align="justify"> The embedding layer is the first layer of a RAG model, and it contains an embedding model that is trained on a massive data set of text and code. Embedding layer is responsible for processing text from document and converting it into meaningful vector embeddings using different chunking strategies and embedding models. This layer is an important part of RAG models because it allows our system to understand the meaning of the text that it is processing and understand its semantic relationship to the query. The embedding layer generates embeddings for our text corpus and allows the RAG model to understand the meaning of the query and to generate a relevant and informative response.<br>

#### **1.1. Document Processing:**
The PDF document will be processed, cleaned, and chunked into smaller parts for embedding.<br>

#### **1.2. Chunking Strategy:**
Several chunking strategies will be tested to find the optimal method that balances the trade-off between chunk size and information retention.<br>

#### **1.3. Embedding Models:**
Embedding models from Hugging face will be evaluated. Here we focus on selecting the model that provides the most accurate, reliable and contextually relevant embeddings for the chunks.<br>

### 2. Search and Rank Layer:
The next layer or second layer is the search and rank or the re-rank layer. Search layer focuses on embedding the queries, performing a vector search using ChromaDB, implementing cache mechanism, and applying re-ranking to retrieve the most relevant information or responses. This layer is an essential component of RAG because it ensures that the retrieved text is accurate, relevant and contextually appropriate. This layer typically consists of two components: A search component that uses various techniques to retrieve relevant documents from the knowledge base and a re-rank component that uses a variety of techniques to re-rank the retrieved documents to produce the most relevant results.<br>

#### **2.1. Query Design:**
In search and rank layer some queries will be created based on the content of the policy document. These queries will be designed to test the system’s ability to retrieve relevant information.<br>

#### **2.2. Query Embedding and Search:**
Queries will be embedded and searched against the ChromaDB vector database. The search results will be ranked, and the top results will be selected for further processing.<br>

#### **2.3. Cache Mechanism:**
A Cache mechanism will be implemented to store and retrieve frequently accessed queries efficiently.<br>

#### **2.4. Re-Ranking:**
The search results will be re-ranked using cross-encoding models from Hugging Face to ensure the most relevant chunks are selected.<br>

### 3. Generation Layer:
The generation layer is the last layer of a RAG model which consists of a foundation large language model that is trained on a massive data set of text and code. Generation layer generates the final answers using a comprehensive prompt designed to leverage the retrieved chunks and optimize response quality.<br>

#### **3.1. Prompt Design:**
A detailed and exhaustive prompt will be created to guide Large Language Model (LLM) in generating the responses.<br>

#### **3.2. Few-Shot Examples:**
To improve the quality of the generated answers, few-shot examples may be included in the prompt.<br>

#### **3.3. Final Output:**
The system will generate a final answer for each query based on the retrieved information.<div/>

## Testing and Evaluation:
<div align="justify"> The system will be tested against the 3 self-designed queries and both search results and final generated answers will be evaluated.<div/>

## Challenges Faced:

#### 1. Text Processing Complexity:
<div align="justify"> Handling large and complex structure of pdf document during text extraction and cleaning is not an easy task, particularly in maintaining context within chunks, determining optimal chunk size and structure for embedding.<br>

#### 2. Model Selection:
Performance and accuracy also depend on the model. Choosing the best embedding and cross-encoder models required extensive testing to balance computational efficiency with retrieval quality.<br>

#### 3. Prompt Design:
Creating an effective prompt that guides LLM to generate accurate and context specific answers required significant testing and improvement.<br>

#### 4. System Performance:
While processing large documents, it is difficult to ensure that the system remains efficient and scalable.<div/>

## Conclusion:
<div align="justify"> This project aims to develop a comprehensive and robust generative search system using RAG pipeline. The final system is expected to accurately answer complex queries from a pdf document, which can demonstrate the power of AI in automating and enhancing information retrieval.<div/>

## Technologies Used:
- Python, version 3
- Hugging face
- Transformers
- ChromaDB
- OpenAI API
- LLM
- Google Colab

## References:
- Python documentations
- Hugging face documentations
- Stack Overflow
- OpenAI
- Kaggle
- Gen AI Articles
- LLM Articles

## Contact Information:
Created by https://github.com/Erkhanal - feel free to contact!
