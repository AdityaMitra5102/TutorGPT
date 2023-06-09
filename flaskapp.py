#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
from flask import *
from ingest import *
import subprocess
import wget
currentpath="."
import webbrowser
import urllib

app=Flask(__name__)

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
from constants import CHROMA_SETTINGS

def load_model():
	global qa
	path=currentpath+"/"+model_path
	check_file=os.path.isfile(path)
	if not check_file:
		os.system("curl https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin -o ./models/ggml-gpt4all-j-v1.3-groovy.bin")
	embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
	db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
	retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
	match model_type:
		case "LlamaCpp":
			llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, verbose=False)
		case "GPT4All":
			llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', verbose=False)
		case _default:
			print(f"Model {model_type} not supported!")
			exit;
	qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= True)
	

@app.route("/")
def home():
	return render_template('index.html')

@app.route("/upload", methods=["GET", "POST"])
def upload():
	files=request.files.getlist("file")
	for file in files:
		file.save(currentpath+"/source_documents/"+file.filename)
		print(file.filename, "UPLOADED")
	subprocess.run(["python", "ingest.py"], capture_output=True)
	load_model()
	return "uploaded"
    
@app.route("/query", methods=["GET","POST"])
def askqs():
	global qa
	query=request.form['query']
	print("Question: ",query)
	res = qa(query)
	answer, docs = res['result'], res['source_documents']
	print("Answer: ",answer)
	x=answer+"\n\nSources:"
	#for document in docs:
	#	x=x+"\n> " + document.metadata["source"] + ":"
	#	x=x+document.page_content
	return x


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

load_model()

if __name__ == "__main__":
	webbrowser.open('http://localhost:5000')
	app.run(host='0.0.0.0', port=5000)