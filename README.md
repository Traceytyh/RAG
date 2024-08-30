# RAG
Comprehensive RAG process from pretraining to evaluation

## Useful links: 
https://huggingface.co/learn/cookbook/en/advanced_rag
https://www.kaggle.com/code/vbookshelf/kagglebot-gemma-7b-it-rag-w-few-shot-prompting#19--Rerank-(reorder)-the-search-results
https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb?ref=blog.langchain.dev
https://python.langchain.com/v0.1/docs/use_cases/question_answering/chat_history/

# RAG Overview
RAG can be split into pre and on production. In pre-production, document loaders are used to prepare the input data which is then broken down into smaller chunks to fit in the embedding model’s context window. It could then be indexed in a vector database. Based on the query’s embedding, a retriever would pick out the necessary information from the database which will be used as the context to create the prompt. An answer would then be generated from the prompt using an LLM.  
## Pre-production
### Text Splitting
Ideally, text chunks have similar semantic meanings and sizes of tokens suitable for the model. A common way is to split by character with overlaps between chunks to keep the context. RecursiveCharacterTextSplitter splits the documents by the specified 'separators' parameter and get further chunked to the tokenizer's maximum sequence length if it is still large.
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    MARKDOWN_SEPARATORS = [
	"\n#{1,6} ",
	"```\n",
	"\n\\*\\*\\*+\n",
	 "\n---+\n",
	"\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
    	AutoTokenizer.from_pretrained(tokenizer_name),
    	chunk_size=AutoTokenizer.from_pretrained(tokenizer_name).max_seq_length, # The maximum number of characters in a chunk: we selected this value arbitrarily
	    chunk_overlap=int(chunk_size / 10),  # The number of characters to overlap between chunks
   	 add_start_index=True,  # If `True`, includes chunk's start index in metadata
   	 strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
   	 separators=MARKDOWN_SEPARATORS,
	)

### Embedding model
Subsequently, both the documents and queries get embedded. Selecting sentence transformers model for embeddings would capture more semantic meanings and works well for large-scale data retrieval. Caching the embeddings in vector stores allows for similarity searching and quick access without needing re-computation. Selecting a model that was trained on both codes and natural language is necessary as the semantic similarities of typical English sentences differ from codes, yet the query is in natural language. Ideally, two separate models should be trained on the questions and answers. However, this would require a large dataset of question answer pairs which are not available.
	from langchain.embeddings import HuggingFace
 	Embeddingsembedding_model = HuggingFaceEmbeddings(
    		model_name='./mlm_collator_code/model_tokenizer', # Use pre-trained model on corpus
    		multi_process=True,
    		model_kwargs={"device": "cuda"},
   		 encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
		)


Depending on the length of the metadata, 512 input token length was found to be insufficient. Try using larger models. 
### Vector Database/Store and Retriever
There are vector databases which involve storing, retrieving and indexing the embeddings. They are able to perform more complex searches on the data, but vectorstores should be sufficient for our use case. Various vector store options include FAISS, Pinecone, Lance, Chroma and Deeplake. Some retrievers make use of similarity scores, top k, reciprocal rank fusion etc to filter the data. The query should be embedded too. 
docs_processed: RecursiveCharacterTextSplitter(...).split_documents(docs)
embedding_model: HuggingFaceEmbeddings(...) # Use own pre-trained model

	from langchain.vectorstores import FAISS

	vectorstore = FAISS.from_documents(
    		docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
		)
	retriever = vectorstore.as_retriever()	

OR CHROMA

from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=docs_processed, embedding=embedding_model)
retriever = vectorstore.as_retriever()
The selection of retrievers would be dependent on the type of query asked, type of data and the expected size output chunk. Indexing then prevents duplication and modification of unchanged contents, should there be real-time updates of documents fed into the vector store. The transformer model/LLM could then use flash attention. It increases the speed as the keys, queries and values are loaded at once rather than a back-and-forth computation. FAISS and Chroma are rather easy to use - comparison of vectorstores can be found here.
## In Production
### Top k documents
Given the experimental stage of this, generating labelled data containing queries, retrieved documents and outputs was not feasible. Hence, to improve the model slightly, a retrieval model was used for reranking of the retrieved documents as they use finer query and document interactions. There may be hallucinations if irrelevant documents are being extracted. 
retrieved_docs = vectorstore.similarity_search(query=user_query, k=5)

from ragatouille import RAGPretrainedModel
RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
relevant_docs = [doc["content"] for doc in relevant_docs]
relevant_docs = relevant_docs[:num_docs_final]
## Further improvements
### Pre-training model - Masked Language Modeling
Common methods to pre-train a model include Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) - for Question Answering, Causal Language Modeling (CLM) - for text generation. Other text corruption strategies also include token masking, deletion, infilling and document rotation. Of these, MLM is better for a model’s contextual understanding with bidirectional learning. Input data like command logs may not have the clearest sentence separators for NSP as a pre-training method. 
•	Tokenizer can first be trained on new corpus (deterministic update in splitting)
tokenizer = tokenizer.train_new_from_iterator([doc.page_content for doc in docs_processed], tokenizer.vocab_size)
•	Datacollator can be used for the data preparation which takes mlm_probability. Of this mlm_probability, 80% of it will be masked while 10% will be replaced with random words in the vocabulary and other 10% unchanged.
•	Data can be packed as well (Packing and Splitting - data preparation for LLMs)
### Prompt engineering
A structured prompt and query were fed into the reader LLM. The prompt had to be clear, specific, provide the appropriate context from the retriever, examples of expected output and a breakdown of the given task. Given a limited input window size, we had to make the prompt as concise as possible. Due to limited question-answer input data, having a well-trained dual encoder for questions and answers is not feasible. However, utilizing these examples as few shot prompts significantly improve the output instead. 
### Implementation
1.	Obtain a list of questions, the context and answer output from RAG, and the corrected answer
2.	Save them into a dataframe to be referenced
a.	df_fshot.columns = ['query','gem_context','response', 'corrected_text']
3.	Fit the question, context and corrected answer into the prompt template
4.	Delete prompt, inputs, and generate_ids to create space
def get_response(query_text, context_list):
    
    prompt = f"""<start_of_turn>user
    Context: {df_fshot.loc[0, 'gem_context']}
    Question: {df_fshot.loc[0, 'query']}<end_of_turn>
    <start_of_turn>model
    {df_fshot.loc[0, 'corrected_text']}<end_of_turn>
    <start_of_turn>user
    Context: {df_fshot.loc[5, 'gem_context']}
    Question: {df_fshot.loc[5, 'query']}<end_of_turn>
    <start_of_turn>model
    {df_fshot.loc[5, 'corrected_text']}<end_of_turn>
    <start_of_turn>user
    Context: {df_fshot.loc[6, 'gem_context']}
    Question: {df_fshot.loc[6, 'query']}<end_of_turn>
    <start_of_turn>model
    {df_fshot.loc[6, 'corrected_text']}<end_of_turn>
    <start_of_turn>user
    Use the context only to answer the question concisely. 
    
    Context: {context_list}
    Question: {query_text}<end_of_turn>
    <start_of_turn>model
    """    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    # Generate the outputs from prompt
    generate_ids = gemma_model.generate(**inputs, max_new_tokens=768)
    # Decode the generated output
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]

    # Clear the memory to create space
    del prompt
    del inputs
    del generate_ids
    torch.cuda.empty_cache() 
    gc.collect()
    
    return response
However, with various contexts for different queries fed into the prompt, one may risk having hallucinations and inaccuracies. 
Advanced Implementation
Instead of having a fixed set of examples, an example_selector could be used to select examples by their: 
•	Semantic similarities
o	from langchain_core.example_selectors import SemanticSimilarityExampleSelector
•	Length
o	from langchain_core.example_selectors import LengthBasedExampleSelector
•	n-gram overlap
o	from langchain_core.example_selectors import NGramOverlapExampleSelector
•	Diversity and relevance 
o	from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
The list of examples are embedded into a vectorclass and fed into FewShotPromptTemplate along with the example selector to generate the prompt. 
### Chat History
Langchain chain
As users may make reference to previous messages, using langchain to incorporate the chat history is needed. The output of huggingface's pipeline 'text generation' and langchain's ConversationBufferMemory would contain the entire text history. However, parsing it directly into the model is unfavorable due to limited input. Instead, with a huggingface token, the huggingface model could be initialized as a model in langchain which has MessageHistory which tracks the chat using a conversation ID. Chains enable us to combine multiple components into a single, coherent application. 
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token = token
)
chat_model = ChatHuggingFace(llm = llm)

 


Langchain_core.runnables.history.RunnableWithMessageHistory: Creates a runnable where conversation id and user id are given to constantly track the messages and utilise it in the prompt
Langchain_core.chat_history.BaseChatMessageHistory : retrieves the old messages from the storage path and session id as a json file and runs through a dictionary which translates the messages.
Langchain_core.messages.ai.AIMessage : message from the model (response to prompt)
BaseMessage:  contains the string of message and additional fields like ID, name, metadata.
BaseRetriever: Runnable interface that returns list of retrieved documents (can use invoke, ainvoke (async), batch, abatch)
Langchain.chains.combine_documents.stuff.create_stuff_documents_chain : Creates chain for passing list of document into model (has llm, prompt template (contain {context}), document prompt (format docs into string)) – documents extracted in manner which fits into the prompt’s context and into the model
Langchain.chains.retrieval.create_retrieval_chain : input retriever and chombine_docs_chain . First retrieve necessary documents then into combine_docs_chain to be prepared for parsing into model
### Langchain ConversationSummaryBufferMemory
Aside from using the chain which stores the entire chat history, to paraphrase the query, a summary of chat history could be used. Previous k conversations could be retained with earlier conversations summarised to the token limit. (convo_summary_mem_buff). 
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=40),
    verbose=True,
)
conversation_with_summary.predict(input="Hi, what's up?") 
Document analysis
Understand the types of files within your corpus to select the type of embedding model (if it is mostly NL/code etc).
import os
def get_file_extensions(root_dir):
    extensions = set()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            _, ext = os.path.splitext(file)
            if ext:  # Check if the file has an extension
                extensions.add(ext.lower())
    return extensions

root_dir = "./the-algorithm" #insert own directory 
extensions = get_file_extensions(root_dir)
print("Unique file extensions found in the directory:")
print(extensions)
One could analyse the distribution of function lengths by parsing .py, .java, .cpp, .rs documents into tree-sitter which obtain nodes (package declarations, class declarations etc,).  It was confirmed that the embedding model maximum input length was sufficient to accommodate the functions.
import tree_sitter_python, tree_sitter_rust, tree_sitter_java, tree_sitter_cpp 
import os
import re
from tree_sitter import Language, Parser
import matplotlib.pyplot as plt
import numpy as np

# Define the root directory and extensions of interest
root_dir = "./the-algorithm"
extensions = {'.py', '.java', '.cpp', '.rs'}

# Load the parsers
PY_LANGUAGE = Language(tree_sitter_python.language())
JAVA_LANGUAGE = Language(tree_sitter_java.language())
CPP_LANGUAGE = Language(tree_sitter_cpp.language())
RUST_LANGUAGE = Language(tree_sitter_rust.language())

LANGUAGE_MAP = {
    '.py': PY_LANGUAGE,
    '.java': JAVA_LANGUAGE,
    '.cpp': CPP_LANGUAGE,
    '.rs': RUST_LANGUAGE,
}

def get_function_lengths(file_path, language):
    parser = Parser(language)
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    tree = parser.parse(bytes(code, 'utf-8'))
    root_node = tree.root_node

    function_lengths=[]
    for child in root_node.children:
        function_lengths.append(len(child.text))

    return function_lengths

function_lengths = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        ext = os.path.splitext(file)[1]
        if ext in extensions:
            language = LANGUAGE_MAP[ext]
            file_path = os.path.join(dirpath, file)
            try:
                function_lengths.extend(get_function_lengths(file_path, language))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

average_length = sum(function_lengths) / len(function_lengths)
print(f"Average function length: {average_length}")
print("Maximum length: ", max(function_lengths))
function_lengths.sort()
print("Median is: " + str(function_lengths[len(function_lengths)//2])) 

import math
plt.hist(function_lengths, bins=round(math.sqrt(len(function_lengths))))
plt.xlabel('Function lengths (No. of characters)')
plt.ylabel('Frequency')
plt.title('Frequency vs. Function Lengths')
plt.grid(True)
plt.show()
Analysing the retrieval process
If the question uses a keyword that could be used to extract specific documents directly, one could manually compare the cosine similarities of the relevant documents with those extracted. The vector store has minimal impact on the retrieval. Should the wrong documents get retrieved, one should change the embedding model selected or preprocess the input documents for better embedding. 
# EXTRACTING RELEVANT DOCUMENTS
relevant_ids = []
content = [doc.page_content for doc in docs_processed]
text = 'favCountParams' # an example of a keyword
for ind, con in enumerate(tqdm(content)):
    if text in con:
        relevant_ids.append(ind) # see the index of relevant documents
print('Should have extracted ', [docs_processed[id].metadata for id in relevant_ids], ' for favCountParams question')

# OBTAINING COSINE SIMILARITIES
from sklearn.metrics.pairwise import cosine_similarity
doc_embedding = {}
similarity = []
retrieved_docs.extend([docs_processed[id].metadata for id in relevant_ids]) #append the relevant documents
for ind, item in enumerate(retrieved_docs):
    embedding = embedding_model.embed_query(item.page_content)
    doc_embedding.update({ind : embedding})
    similarity.append(cosine_similarity([query_vector, embedding])[0][1])
print(similarity) # similarity coefficients of the retrieved documents followed by those relevant documents
## Post-process output
If the output consistently gives specific redundant statements, it could be manually filtered: 
Original output:
"The provided context does not contain any specific details about ... Therefore, I am unable to provide an explanation or description for it based on this given text. Please refer to appropriate documentation related to your project where... might have been defined."
Processed output:
"Sorry, that information is not available."
## Evaluation experiments
Input a series of questions and check the output for unwanted phrases (eg. 'sure!' or 'From the context,'), hallucinations, and inaccurate information. Record the time taken for output to be generated. 
5.	Robustness 
a.	Input the same question with a slight grammatical error - varying output?
b.	Input a long question - does the notebook crash? - length of context parsed and few-shot examples may need to be altered
6.	Experiment the minimum context chunk size for accurate answer (typically 2-3) and the need for specifying the role of the system
## Handling images and tables for RAG
Options:
1) Embed images and text using CLIP, pass raw images and text into multimodal LLM 
2) Summarize images and tables, embed all these as text, pass text into LLM
3) Summarize images and tables, embed summaries but with reference to image, pass text and images into multimodal LLM
Considerations for selecting option 3:
1) CLIP embeds by using a text encoder to pair images into a class which will not be represented with 3GPP data
2) Summaries would be vague, despite being used to express complex ideas 
3) Summaries are less important (only needed to identify relevant images for the RAG) as multimodal LLM are used
 
### Extracting Images, Texts, Tables
1) Unstructured (GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.) parses the pdf to be represented as elements. Tables are saved as html to retain its formatting. Images are saved into an output directory. 
2) Camelot (GitHub - atlanhq/camelot: Camelot: PDF Table Extraction for Humans) saves tables into csv file. OCR/Scjda/Mathpix can be used to extract diagrams from pdfs. 
3) BeautifulSoup can be used to extract images, figure captions, and texts etc from HTML (urllib to parse website)
from bs4 import BeautifulSoup
from urllib.request import urlopen
weblink = "##insert link"
page = urlopen(weblink)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
figcaption = soup.find_all("figcaption")
txt = soup.get_text()
images = soup.find_all("img")
### Directly embedding images
1) CLIP (CLIP (huggingface.co))
2) ImageBind (GitHub - facebookresearch/ImageBind: ImageBind One Embedding Space to Bind Them All)
Summarizing Images and Tables before embedding
1) Multimodals eg. LlaVa can summarize image objects 
Using Llava to summarize some images in 3gpp.org/technologies/ee-article without pertaining gave some hallucinations, wrong information, trivial information, and vague descriptions. However, it showed some abilities to draw from text within the image. First they will be fed through a processor which combines both text (tokenizer) and image processor (resizing, normalizing and conversion to tensors).
from PIL import Image
import requests
import os
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt = "USER: <image>\nDescribe the content of the image. What about telecommunication does it illustrate? ASSISTANT:"
for images in os.listdir('pdf/images/'):
    try:
        image = Image.open('pdf/images/'+images)
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        print("Image :", images)
        generate_ids = model.generate(**inputs, max_new_tokens=30)
        print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    except:
        #skip the .svg files
        pass 
Image	 	 	 	 
	 	 
Summary with contextualized prompt:
Describe the content of the image. 
What about telecommunication does it illustrate?	The image features a three-dimensional representation of a cellular network, with multiple red dots representing cell towers. These dots are arranged in 	The image illustrates a complex network of interconnected cells, each with a red dot. This visual representation of the network is likely used to illustrate the 	The image illustrates the structure and function of a cell, specifically a Uranus cell. It shows the cell's components, such as the	The image displays a mathematical equation involving a variable, specifically the letter "d". This equation is likely related to telecommunication, as the letter " 	The image illustrates a complex telecommunication network, with multiple layers and connections between different components. The network is composed of various nodes, including a 	The image illustrates a graph showing the relationship between the power consumption and the CPU utilization of a computer. The graph is a line graph with a 
Summary with generic prompt:
Describe the content of the image. 
What is it illustrating?	The image features a three-dimensional representation of a cell, with a focus on the cell membrane. The cell membrane is depicted as a
	The image illustrates a complex network of interconnected cells, with each cell having a unique identifier. The cells are arranged in a three-dimensional structure
 	The image illustrates a diagram of a cell, focusing on the cell's components and their interactions. The diagram features a cell with a large	The image displays a mathematical equation written in a white color on a black background. The equation is a combination of letters and numbers, possibly representing a formula	The image illustrates a diagram or a flowchart that represents a process or a system. The diagram consists of multiple boxes, each containing a letter or	The image displays a graph that illustrates the relationship between the power consumption of a computer and the number of CPUs (central processing units) utilized


Depending on the data input, figure captions are typically a better description of the image and could be saved as the image's summary or for fine-tuning the model. Alternatively, one could include the image if it is near a retrieved chunk (requires the indexing in the vectorstore to have the same sequence as the document). Images such as logos which are deemed irrelevant (based on use case) could be removed from the dataset. 
Image observations (while looking through various 3gpp / other documents):
•	Figure types include:
o	Graphs describing relationship between two parameters
o	Diagrams to explain system function/relationships between components
o	Formulas
o	Infographics (containing a lot of text)
o	Timelines
o	Logos
o	Images for aesthetic purposes (not informative)
•	Not all figure numbers are being referenced within the text despite being explained
•	Figures may not be labelled
•	Images do not have very high resolution (hard to read text)
•	Image captions (src from websites) are usually not informative
2) Images can be encoded into eg. base64 before being fed into LLM such as  “gpt-4-vision-preview”

Multimodal LLM for answer synthesis
Llava 1.6 is still trained on single image input. Figure captions could be used to fine-tune an LLM by running on an A100 GPU with preferably 40-80GB memory/using parallelism.

