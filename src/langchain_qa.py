import os
from typing import Dict, List
import json
import platform
import yaml
import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import RequestsHttpConnection

from langchain.chains.question_answering import load_qa_chain
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.schema import Document

import sys

from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.vectorstores import OpenSearchVectorSearch
##
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.load.dump import dumps



sys.path.append("../")

with open("../src/config.yml", "r") as file:
    config = yaml.safe_load(file)


def validate_environment():
    assert platform.python_version() >= "3.10"

def bedrock_endpoint_embeddings():
    region = os.environ.get("AWS_REGION")
    session = boto3.Session() 
    boto3_bedrock = session.client('bedrock-runtime', region, endpoint_url='https://bedrock-runtime.'+region+'.amazonaws.com')
    return BedrockEmbeddings(client=boto3_bedrock)
    
def amazon_opensearch_docsearch(aos_config, docs, embeddings):
    _aos_host = aos_config["aos_host"]
    port = 443
    region = os.environ.get("AWS_REGION")  # e.g. us-west-1  # e.g. us-west-1

    service = "aoss"
    credentials = boto3.Session().get_credentials()

    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token,
    )

    docsearch = OpenSearchVectorSearch.from_texts(
        texts=[d.page_content for d in docs],
        embedding=embeddings,
        metadatas=[d.metadata for d in docs],
        opensearch_url=[{"host": _aos_host, "port": port}],
        index_name=aos_config["aos_index"],
        http_auth=awsauth,
        use_ssl=True,
        pre_delete_index=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=100000,
    )
    return docsearch

def create_vector_store(model__, embed_model, docs, aos_config):
    if embed_model == "amazon.titan-embed-text-v1":
        embeddings = bedrock_endpoint_embeddings()
    else:
        assert False

    return amazon_opensearch_docsearch(
        aos_config=aos_config, docs=docs, embeddings=embeddings
    )


def amazon_kendra_retriever(kendra_config):
    return boto3.client("kendra")

def bedrock_text_endpoint(endpoint_name):
    """anthropic.claude-v2"""
    region = os.environ.get("AWS_REGION")
    session = boto3.Session() 
    boto3_bedrock = session.client('bedrock-runtime', region, endpoint_url='https://bedrock-runtime.'+region+'.amazonaws.com')
    return Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={"max_tokens_to_sample": 200})

def chain_qa(llm, verbose=False):
    return load_qa_chain(llm, chain_type="stuff", verbose=verbose)

def search_and_answer(store, chain, query, k=1, doc_source_contains=None):

    if isinstance(store, OpenSearchVectorSearch):
        docs = store.similarity_search(
            query,
            k=k,
            # include_metadata=False,
            verbose=False,
        )
    elif store.__class__.__name__ == "kendra":
        response = store.retrieve(IndexId=config["kendra"]["index_id"], QueryText=query)
        docs = [Document(page_content=r["Content"]) for r in response["ResultItems"]]
    else:
        assert False, f"Unknown doc store {type(store)}"

    response = chain.run(input_documents=docs, question=query)
    return {"response": response, "docs": docs}
