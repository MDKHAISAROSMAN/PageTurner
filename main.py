import json
from typing import List
import os

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


file_path = 'test.pdf'

def partition_document(file_path:str):
    """Extract elements from PDF using unstructured"""
    print(f"Partitioning Document {file_path}")

    elements = partition_pdf(
        filename= file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    ) 
    print(f"Extracted {len(elements)} elements")
    return elements

def create_chunk_by_title(elements):
    """Creating chunks based on title"""
    print("creating smart chunks...")
    chunks = chunk_by_title(
        elements,
        max_characters = 3000,
        new_after_n_chars = 2400, #Try to start new chunk after 2400 characters
        combine_text_under_n_chars = 500  #merge tiny chunks under 500 characters with neighbours
    )
    print(f"Created {len(chunks)} chunks")
    return chunks

def seperate_content_type(chunk):
    """Analyze what types of elements are in the chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }

    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)

            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)

    content_data['types'] = list(set(content_data['types']))
    return content_data


def create_ai_enhanced_summary(text, tables, images):
    """Create AI summary for mixed content"""
    try:
        llm = ChatOllama(model="qwen2.5vl:7b")
        prompt_text = f"""You are creating a searchable description for document content retrieval. Only reply with the description and nothing else.

CONTENT TO ANALYZE:
TEXT CONTENT:
{text}

YOUR TASK:
YOUR TASK:

Generate a comprehensive, searchable description of the provided document that covers:

1. Key facts, numbers, statistics, dates, and data points from text and tables.
2. Main topics, concepts, definitions, and arguments discussed.
3. Important entities, names, organizations, technologies, and terminology.
4. Questions that this document could answer.
5. Visual content analysis, including charts, diagrams, tables, images, and their relationships.
6. Relationships between concepts, cause-and-effect explanations, and dependencies.
7. Alternative search terms, synonyms, abbreviations, and related keywords users might use.
8. Important conclusions, recommendations, and takeaways.

Make the description detailed, information-dense, and optimized for semantic search and retrieval. Prioritize factual accuracy and findability over brevity.

Preserve exact numerical values, technical terminology, and named entities. Do not invent information or infer unsupported facts. Explicitly indicate when information is missing or unclear.
"""

        if tables:
            prompt_text += "\nTables:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i + 1}:\n{table}\n"

        message_content = [{"type": "text", "text": prompt_text}]

        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        return response.content if hasattr(response, "content") else response

    except Exception as e:
        print(f"AI summary Failed: {e}")

        summary = f"{text[:300]}..."

        if tables:
            summary += f" Contains {len(tables)} table(s)."
        if images:
            summary += f" Contains {len(images)} image(s)."

        return summary


def summarize_chunks(chunks):
    """Process all chunks with AI summaries"""
    print("Processing chunks with AI summaries...")

    langchain_documents = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"Processing Chunk {current_chunk}/{total_chunks}")

        content_data = seperate_content_type(chunk)

        print(f"Types found {content_data['types']}")
        print(f"Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

        if content_data['tables'] or content_data['images']:
            print("Creating AI summary for mixed data")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'],
                    content_data['images']
                )
            except Exception as e:
                print(f"AI summary failed: {e}")
                enhanced_content = content_data['text']
        else:
            print("Using raw text (no tables/images)")
            enhanced_content = content_data['text']

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images": content_data['images']
                })
            }
        )

        langchain_documents.append(doc)
        print(f"Processed {len(langchain_documents)} chunks")

    return langchain_documents

def export_chunk_to_json(chunks,filename='clean_chunks.json'):
    """Export prossed chunks to clean JSON"""
    export_data = []
    for i ,doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i+1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content","{}"))
            }
        }
        export_data.append(chunk_data)

    with open(filename,'w',encoding="utf-8") as f:
        json.dump(export_data,f,indent=2,ensure_ascii=False)
    return export_data

def create_vector_store(documents,persist_dir="db2/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embedding and storing in ChromaDB")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    print("----Creating Vector store-----")
    vectorestore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_metadata = {"hnsw:space":"cosine"}
    )
    print("---Finished creating Vector store----")
    return vectorestore

def generate_final_answer(chunks,query):
    """Generate final answer using multimodal content"""

    try:
        llm = ChatOllama(model="qwen2.5vl:7b")
        prompt_text = f"""Based on the following documents, Please answer this question: {query}
        Content to Analyze:"""

        for i ,chunk in enumerate(chunks):
            prompt_text += f"---Document {i+1}"

            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])

            raw_text = original_data.get("raw_text","")
            if raw_text:
                prompt_text += f"TEXT:\n{raw_text}\n\n"

            tables_html = original_data.get("tables_html",[])
            if tables_html:
                prompt_text += "TABLES\n"
                for j,table in enumerate(tables_html):
                    prompt_text += f"Table {j+1}:\n{table}\n\n"
            prompt_text += "\n"
        
        prompt_text  += """Please provide a clear and comprehensive answer using the text,tables and images above, if the document does not contain sufficient content then respod by sayig i dont have enough data to answer and just give the answer and dont start by here is the answer etc"""
        
        message_content = [{'type':'text','text':prompt_text}]
        
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata['original_content'])
                images_base64 = original_data.get("images_base64",[])

                for image_base64 in images_base64:
                    message_content.append({
                        "type":"image_url",
                        "image_url":{"url":f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        return response.content if hasattr(response,"content") else response

    except Exception as e:
        print(f"Answer Generation Failed: {e}")
        return "Sorry, I encountered an error while generating the answer."

def rewrite_query(query,chat_history):
    message = [SystemMessage(content="Given the chat history, rewrite the new question so that it can be standalone and searchable and requires no additional history and only reply with the question nothing else")]+chat_history+[HumanMessage(content=f"new question; {query}")]
    llm = ChatOllama(model="qwen2.5vl:7b")
    result = llm.invoke(message)
    new_question = result.content if hasattr(result,'content') else result 
    return new_question


def chat(db):
    chat_history = []
    while True:
        query = input("Enter the query (Enter quit to exit): ")

        if query.lower() == 'quit':
            print("GoodBye!")
            break

        if chat_history:
            query = rewrite_query(query,chat_history)
        
        retriver = db.as_retriever(search_kwargs={'k':3})
        ret_chunks = retriver.invoke(query)
        export_chunk_to_json(chunks=ret_chunks,filename="ret_results.json")
        final_answer = generate_final_answer(ret_chunks,query)
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=final_answer))
        print(final_answer)

def main():
    # comment out to use already made db
    #elements = partition_document(file_path)
    #chunks = create_chunk_by_title(elements)
    #processed_chunks = summarize_chunks(chunks)
    #db = create_vector_store(processed_chunks)
    #uncomment to use already built db
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db =  Chroma(
    persist_directory="db2/chroma_db",  # Path to saved folder
    embedding_function=embeddings,
    )
    chat(db)


if __name__ == '__main__':
    if not file_path.lower().endswith('.pdf'):
        print("Error: Please send PDF file only!")
    else:
        main()