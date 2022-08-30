from flask import Flask, render_template, request
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from haystack.utils import print_documents
from haystack.pipelines import DocumentSearchPipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever


app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return render_template('./index.html')

@app.route('/aboutus',methods=['GET'])
def About():
    return render_template('./Aboutus.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        query = request.form['query']
        new_document_store = FAISSDocumentStore.load("./faiss.index")
        new_retriever = DensePassageRetriever.load("./retriever.pt", document_store=new_document_store)
        
        p_retrieval = DocumentSearchPipeline(new_retriever)
        
        res = p_retrieval.run(query=query, params={"Retriever": {"top_k": 5}})
                       
        return render_template('./index.html',list = res,question=query)
    else:
        return render_template('./index.html')
     

if __name__=="__main__":
    app.run(debug=True)