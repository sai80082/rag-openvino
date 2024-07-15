from pathlib import Path
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.postprocessor.openvino_rerank import OpenVINORerank
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.llms.openvino import OpenVINOLLM

from llm_config import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_RERANK_MODELS,
    SUPPORTED_LLM_MODELS,
)


model_language = "English"
llm_model_id = "gemma-2b-it"
embedding_model_id = "bge-small-en-v1.5"
rerank_model_id = "bge-reranker-large"


embedding_device = "CPU"
rerank_device = "CPU"
llm_device = "CPU"

int8_model_dir = Path(llm_model_id) / "INT8_compressed_weights"
int8_weights = int8_model_dir / "openvino_model.bin"

for precision, compressed_weights in zip([8], [int8_weights]):
    if compressed_weights.exists():
        print(f"Size of model with INT{precision} compressed weights is {compressed_weights.stat().st_size / 1024 / 1024:.2f} MB")

embedding = OpenVINOEmbedding(folder_name=embedding_model_id, device=embedding_device)
reranker = OpenVINORerank(model=rerank_model_id, device=rerank_device, top_n=3)
model_dir = int8_model_dir
print(f"Loading model from {model_dir}")

ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

# On a GPU device a model is executed in FP16 precision. For red-pajama-3b-chat model there known accuracy
# issues caused by this, which we avoid by setting precision hint to "f32".

llm = OpenVINOLLM(
    model_name=str(model_dir),
    tokenizer_name=str(model_dir),
    context_window=2048,
    max_new_tokens=1024,
    model_kwargs={"ov_config": ov_config, "trust_remote_code": True},
    generate_kwargs={"temperature": 0.2, "top_k": 50, "top_p": 0.95},
    device_map=llm_device,
)

llm_model_configuration = SUPPORTED_LLM_MODELS[model_language][llm_model_id]

d = embedding._model.request.outputs[0].get_partial_shape()[2].get_length()
faiss_index = faiss.IndexFlatL2(d)
Settings.embed_model = embedding
Settings.llm = llm
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)







from langchain.prompts import PromptTemplate
from llama_index.core.prompts import LangchainPromptTemplate

rag_prompt_template = llm_model_configuration["rag_prompt_template"]
stop_tokens = llm_model_configuration.get("stop_tokens")

langchain_prompt = PromptTemplate.from_template(rag_prompt_template)

lc_prompt_tmpl = LangchainPromptTemplate(
    template=langchain_prompt,
    template_var_mappings={"query_str": "input", "context_str": "context"},
)



from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from transformers import StoppingCriteria
import torch
import gradio as gr

TEXT_SPLITERS = {
    "SentenceSplitter": SentenceSplitter,
    "RecursiveCharacter": RecursiveCharacterTextSplitter,
}



english_examples = [
    ["what is this document?"],

]

examples =  english_examples


def default_partial_text_processor(partial_text: str, new_text: str):
    """
    helper for updating partially generated answer, used by default

    Params:
      partial_text: text buffer for storing previosly generated text
      new_text: text update for the current step
    Returns:
      updated text string

    """
    partial_text += new_text
    return partial_text


text_processor = llm_model_configuration.get("partial_text_processor", default_partial_text_processor)


class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


if stop_tokens is not None:
    if isinstance(stop_tokens[0], str):
        stop_tokens = llm._tokenizer.convert_tokens_to_ids(stop_tokens)
    stop_tokens = [StopOnTokens(stop_tokens)]


def create_vectordb(doc, spliter_name, chunk_size, chunk_overlap, vector_search_top_k, vector_rerank_top_n, run_rerank):
    """
    Initialize a vector database

    Params:
      doc: orignal documents provided by user
      chunk_size:  size of a single sentence chunk
      chunk_overlap: overlap size between 2 chunks
      vector_search_top_k: Vector search top k
      vector_rerank_top_n: Rerrank top n
      run_rerank: whether to run reranker

    """
    global query_engine
    global index

    if vector_rerank_top_n > vector_search_top_k:
        gr.Warning("Search top k must >= Rerank top n")

    loader = PyMuPDFReader()
    documents = loader.load(file_path=doc.name)
    spliter = TEXT_SPLITERS[spliter_name](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if spliter_name == "RecursiveCharacter":
        spliter = LangchainNodeParser(spliter)
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[spliter],
    )
    if run_rerank:
        reranker.top_n = vector_rerank_top_n
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=vector_search_top_k, node_postprocessors=[reranker])
    else:
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=vector_search_top_k)

    query_engine.update_prompts({"response_synthesizer:text_qa_template": lc_prompt_tmpl})
    return "Vector database is Ready"


def update_retriever(vector_search_top_k, vector_rerank_top_n, run_rerank):
    """
    Update retriever

    Params:
      vector_search_top_k: size of searching results
      vector_rerank_top_n:  size of rerank results
      run_rerank: whether run rerank step

    """
    global query_engine
    global index

    if vector_rerank_top_n > vector_search_top_k:
        gr.Warning("Search top k must >= Rerank top n")

    if run_rerank:
        reranker.top_n = vector_rerank_top_n
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=vector_search_top_k, node_postprocessors=[reranker])
    else:
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=vector_search_top_k)

    query_engine.update_prompts({"response_synthesizer:text_qa_template": lc_prompt_tmpl})


def user(message, history):
    """
    callback function for updating user messages in interface on submit button click

    Params:
      message: current message
      history: conversation history
    Returns:
      None
    """
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def bot(history, temperature, top_p, top_k, repetition_penalty, do_rag):
    """
    callback function for running chatbot on submit button click

    Params:
      history: conversation history
      temperature:  parameter for control the level of creativity in AI-generated text.
                    By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
      top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
      top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
      repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
      do_rag: whether do RAG when generating texts.

    """
    llm.generate_kwargs = dict(
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    if stop_tokens is not None:
        llm._stopping_criteria = StoppingCriteriaList(stop_tokens)

    partial_text = ""
    if do_rag:
        streaming_response = query_engine.query(history[-1][0])
        for new_text in streaming_response.response_gen:
            partial_text = text_processor(partial_text, new_text)
            history[-1][1] = partial_text
            yield history
    else:
        input_text = rag_prompt_template.format(input=history[-1][0], context="")
        streaming_response = llm.stream_complete(input_text)
        for new_text in streaming_response:
            partial_text = text_processor(partial_text, new_text.delta)
            history[-1][1] = partial_text
            yield history


def request_cancel():
    llm._model.request.cancel()


def clear_files():
    return "Vector Store is Not ready"


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    
    gr.Markdown(f"""<center>Legal Advice and Documentation Knowledge Base Chatbot </center>""")
    with gr.Row():
        with gr.Column(scale=1):
            docs = gr.File(
                label="Step 1: Load a PDF file",
                
                file_types=[
                    ".pdf",
                ],
            )
            load_docs = gr.Button("Step 2: Build Vector Store", variant="primary")
            db_argument = gr.Accordion("Vector Store Configuration", open=False)
            with db_argument:
                spliter = gr.Dropdown(
                    ["SentenceSplitter", "RecursiveCharacter"],
                    value="SentenceSplitter",
                    label="Text Spliter",
                    info="Method used to splite the documents",
                    multiselect=False,
                )

                chunk_size = gr.Slider(
                    label="Chunk size",
                    value=200,
                    minimum=50,
                    maximum=2000,
                    step=50,
                    interactive=True,
                    info="Size of sentence chunk",
                )

                chunk_overlap = gr.Slider(
                    label="Chunk overlap",
                    value=20,
                    minimum=0,
                    maximum=400,
                    step=10,
                    interactive=True,
                    info=("Overlap between 2 chunks"),
                )

            vector_store_status = gr.Textbox(
                label="Vector Store Status",
                value="Vector Store is Ready",
                interactive=False,
            )
            do_rag = gr.Checkbox(
                value=True,
                label="RAG is ON",
                interactive=True,
                info="Whether to do RAG for generation",
            )
            with gr.Accordion("Generation Configuration", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.1,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=1.0,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sample from the smallest possible set of tokens whose cumulative probability "
                                    "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k",
                                value=50,
                                minimum=0.0,
                                maximum=200,
                                step=1,
                                interactive=True,
                                info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.1,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=600,
                label="Step 3: Input Query",
            )
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        msg = gr.Textbox(
                            label="QA Message Box",
                            placeholder="Chat Message Box",
                            show_label=False,
                            container=False,
                        )
                with gr.Column():
                    with gr.Row():
                        submit = gr.Button("Submit", variant="primary")
                        stop = gr.Button("Stop")
                        clear = gr.Button("Clear")
            gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")
            retriever_argument = gr.Accordion("Retriever Configuration", open=True)
            with retriever_argument:
                with gr.Row():
                    with gr.Row():
                        do_rerank = gr.Checkbox(
                            value=True,
                            label="Rerank searching result",
                            interactive=True,
                        )
                    with gr.Row():
                        vector_rerank_top_n = gr.Slider(
                            1,
                            10,
                            value=2,
                            step=1,
                            label="Rerank top n",
                            info="Number of rerank results",
                            interactive=True,
                        )
                    with gr.Row():
                        vector_search_top_k = gr.Slider(
                            1,
                            50,
                            value=10,
                            step=1,
                            label="Search top k",
                            info="Search top k must >= Rerank top n",
                            interactive=True,
                        )
    docs.clear(clear_files, outputs=[vector_store_status], queue=False)
    load_docs.click(
        create_vectordb,
        inputs=[docs, spliter, chunk_size, chunk_overlap, vector_search_top_k, vector_rerank_top_n, do_rerank],
        outputs=[vector_store_status],
        queue=False,
    )
    submit_event = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot,
        [chatbot, temperature, top_p, top_k, repetition_penalty, do_rag],
        chatbot,
        queue=True,
    )
    submit_click_event = submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot,
        [chatbot, temperature, top_p, top_k, repetition_penalty, do_rag],
        chatbot,
        queue=True,
    )
    stop.click(
        fn=request_cancel,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    vector_search_top_k.release(
        update_retriever,
        [vector_search_top_k, vector_rerank_top_n, do_rerank],
    )
    vector_rerank_top_n.release(
        update_retriever,
        [vector_search_top_k, vector_rerank_top_n, do_rerank],
    )
    do_rerank.change(
        update_retriever,
        [vector_search_top_k, vector_rerank_top_n, do_rerank],
    )


demo.queue()
# if you are launching remotely, specify server_name and server_port
#  demo.launch(server_name='your server name', server_port='server port in int')
# if you have any issue to launch on your platform, you can pass share=True to launch method:
# demo.launch(share=True)
# it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
demo.launch()

