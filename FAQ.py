import logging
import gradio as gr

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# Creacion de un almacenamiento de documentos simple y en linea
from haystack.document_stores import InMemoryDocumentStore
document_store = InMemoryDocumentStore()

# Uso de embeddings para la creacion del Retriever
from haystack.nodes import EmbeddingRetriever

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
    use_gpu = True,
    scale_score = False,
)

import pandas as pd
df = pd.read_excel("UDEC Preguntas frecuentes.xlsx")

df["Pregunta"] = df["Pregunta"].apply(lambda x: x.strip())

# Embedding de las preguntas, no se realiza embedding de las respuestas, dado que se quiere mapear las preguntas realizadas con las preguntas
questions = list(df["Pregunta"].values)
df["embedding"] = retriever.embed_queries(queries = questions).tolist()
df = df.rename(columns=({'Pregunta':'content'}))

docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)

# Pipeline para realziar preguntas
from haystack.pipelines import FAQPipeline
pipe = FAQPipeline(retriever=retriever)

from haystack.utils import print_answers

#query = "Se tiene algun tipo de beca en la U?"

def responder(entrada):
    prediction = pipe.run(query=entrada, params={"Retriever": {"top_k": 1}})
    answers = prediction["answers"]
    respuesta = ""

    if answers:
        answer = answers[0]
        respuesta = f"{answer.answer}"

    return respuesta

demo2 = gr.Interface(fn = responder,
                   inputs ="text",
                   outputs= "text"
                )

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            pregunta = gr.Textbox(label = "Preguntame")
            responder_btn = gr.Button(value = "Responder")
        with gr.Column():
            respuesta = gr.Textbox(label = "Respuesta")

    responder_btn.click(responder, inputs = pregunta, outputs = respuesta)
    examples = gr.Examples(examples = ['Cuantas sedes tiene la U', 'Que becas tiene la U', 'La universidad es publica?'], inputs=[pregunta])

demo.launch(share = True)
