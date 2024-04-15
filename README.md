# LLM. Procesamiento del lenguaje Natural
Los desafíos desarrollados involucran la implementación de sistemas de procesamiento del lenguaje natural (NLP) utilizando varias herramientas y bibliotecas, como Lang CHain, Pinecone y OpenAI.
1. **Integración de ChatGPT con Lang Chain y OpenAI:**
   Se trata de escribir un programa en Python que envíe solicitudes de entrada a ChatGPT y recupere las respuestas utilizando Lang Chain y la API de OpenAI. Este desafío proporciona un ejemplo de cómo interactuar con ChatGPT utilizando Lang Chain y OpenAI.

2. **Desarrollo de un RAG (Generación de Respuestas Asistida por Recuperación) con una base de datos vectorial en memoria:**
   El objetivo aquí es construir un RAG utilizando una base de datos vectorial en memoria. Se utiliza la biblioteca Lang Chain junto con OpenAI para crear un sistema de respuesta basado en la recuperación de información utilizando modelos de lenguaje de OpenAI y una base de datos vectorial.

3. **Implementación de un RAG utilizando Pinecone:**
   Se emplea Pinecone junto con Lang Chain para construir un RAG (Generación de Respuestas Asistida por Recuperación) utilizando una base de datos vectorial. La tarea implica la carga de documentos de texto, la creación de embeddings utilizando OpenAI, y la búsqueda de similitud de documentos utilizando Pinecone.

## Instalación
Para llevar a cabo nuestras tareas de desarrollo, es fundamental contar con un conjunto específico de paquetes que nos brindan funcionalidades esenciales. 

* **jupyterlab:** Proporciona un entorno interactivo basado en navegador para escribir y ejecutar código en varios lenguajes de programación, incluido Python. Es especialmente útil para la exploración de datos y la creación de notebooks interactivos.

* **openai:** Esta biblioteca nos permite acceder a la API de OpenAI, lo que nos permite integrar modelos de lenguaje avanzados, como GPT, en nuestras aplicaciones.

* **tiktoken:** Es un tokenizador BPE (Byte Pair Encoding) rápido diseñado para su uso con los modelos de OpenAI. Permite convertir texto en secuencias de tokens de manera eficiente, lo que es fundamental para el procesamiento de lenguaje natural. tiktoken es especialmente útil para la tokenización de texto antes de ser procesado por modelos de lenguaje como GPT. Además, ofrece un rendimiento significativamente mejor que otros tokenizadores de código abierto comparables, lo que lo hace ideal para aplicaciones que requieren velocidad y eficiencia en el procesamiento de grandes cantidades de texto. El paquete también proporciona una API documentada y ejemplos de código para su uso en el "OpenAI Cookbook", lo que facilita su implementación en proyectos de NLP.

* **langchain:** Es una biblioteca que facilita la creación y el manejo de cadenas de procesamiento del lenguaje natural (NLP). Ofrece herramientas y utilidades para trabajar con modelos de NLP, procesar texto y más.

* **chromadb** s un cliente Python para interactuar con Chroma, una base de datos que almacena y permite buscar documentos mediante la comparación de vectores de embeddings. Chromadb simplifica el proceso de configuración y uso de Chroma, lo que facilita la creación de aplicaciones de búsqueda y recuperación de documentos. 

* **langchainhub:** Es un componente de Lang Chain que permite acceder y compartir modelos de lenguaje y recursos relacionados con el procesamiento del lenguaje natural desde un hub centralizado.

* **bs4:** Abreviatura de Beautiful Soup 4, es una biblioteca de Python para extraer datos de archivos HTML y XML. Proporciona formas fáciles de navegar, buscar y modificar la estructura del árbol del documento.

* **pinecone-client:** Proporciona una interfaz para interactuar con Pinecone, un servicio para la búsqueda y recuperación de vectores similares a través de bases de datos vectoriales. Se utiliza para trabajar con espacios vectoriales y realizar consultas de similitud.

* **langchain-pinecone:** Es una integración entre Lang Chain y Pinecone, que permite utilizar las capacidades de Pinecone dentro del marco de Lang Chain para tareas relacionadas con el procesamiento del lenguaje natural y la búsqueda de similitud.

* **langchain-community:** Esta biblioteca agrega funcionalidades y componentes desarrollados por la comunidad de Lang Chain, proporcionando una variedad de herramientas y recursos adicionales para el procesamiento del lenguaje natural.

Estos paquetes se agrupan en un archivo requirements.txt, que especifica las dependencias necesarias para el proyecto. 

Para instalar todas las dependencias, simplemente ejecuta el comando:

```
pip install -r requirements.txt
```

Otra parte importante es contar con una cuenta en pinecone. Allí se nos proporcionará un API Key Para el tercer RAG

![](/img/pinecone.png)

* **Puedes crear tu cuenta en este enlace: [https://app.pinecone.io](https://app.pinecone.io)** 

### Programa 1: Usando Python para interactuar con ChatGPT

1. **Configuración del Entorno y Dependencias**:
   - Importar los módulos y librerías necesarios.
   - Establecer la clave de la API de OpenAI como una variable de entorno.

2. **Definir la Plantilla de la Consulta**:
   - Definir una plantilla para la consulta que incluya un marcador de posición para la pregunta.
   - Crear un objeto PromptTemplate con la plantilla y la variable de entrada.

3. **Inicialización del Modelo de Lenguaje (LLM)**:
   - Inicializar una instancia del modelo de lenguaje de OpenAI.

4. **Ejecutar el Programa**:
   - Especificar una pregunta.
   - Pasar la pregunta a través de la cadena del modelo de lenguaje.
   - Imprimir la respuesta.

#### Código

```Python
from langchain.chains import LLMChain
#from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

import os

os.environ["OPENAI_API_KEY"] = "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"



template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI()

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is at the core of Popper's theory of science?"

response = llm_chain.run(question)
print(response)
```

#### Resultado

![](/img/p1.png)

**Pregunta:** *What is at the core of Popper's theory of science?*

**Salida:** *Popper's theory of science is based on the idea of falsification, which is the process of testing and disproving hypotheses. This is in contrast to the traditional view of science as a process of verification or confirmation of hypotheses. At the core of Popper's theory is the concept of falsifiability, which means that for a theory to be considered scientific, it must be possible to test it and potentially prove it wrong. This idea of falsifiability is closely tied to the concept of empirical evidence, which is the evidence gathered through observation and experimentation. According to Popper, scientific theories must be open to being proven wrong through empirical evidence, and this process of constant testing and refinement is what drives scientific progress. In this way, Popper's theory emphasizes the importance of critical thinking and the continuous questioning and revision of scientific ideas.*

### Programa 2: Construir un Sistema RAG (Recuperación con Generación) usando una base de datos vectorial en memoria

1. **Configuración del Entorno y Dependencias**:
   - Importar los módulos y librerías necesarios.
   - Establecer la clave de la API de OpenAI como una variable de entorno.

2. **Cargar Documentos y Dividir el Texto**:
   - Cargar documentos desde una fuente web usando BeautifulSoup.
   - Dividir el texto en fragmentos más pequeños para su procesamiento.

3. **Crear una Base de Datos Vectorial**:
   - Utilizar Chroma para crear una base de datos vectorial a partir de los fragmentos de texto.

4. **Inicialización de la Cadena RAG**:
   - Definir una función para formatear los documentos recuperados.
   - Inicializar la cadena RAG con los componentes del recuperador, la consulta y el modelo de lenguaje.

5. **Ejecutar el Sistema RAG**:
   - Invocar la cadena RAG con una pregunta.
   - Imprimir la respuesta.

#### Código

```Python
import bs4
from langchain import hub
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

os.environ["OPENAI_API_KEY"] = "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"


loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(splits[0])
print(splits[1])

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is Task Decomposition?")

print(response)
```
#### Resultado

![](/img/p2.png)

**Pregunta:** *What is Task Decomposition?*

**Salida:** *Task Decomposition is a technique used to break down complex tasks into smaller and simpler steps. This approach helps agents to plan and execute tasks more efficiently by transforming big tasks into manageable subtasks. Task decomposition can be achieved through prompting techniques like Chain of Thought or Tree of Thoughts, as well as with task-specific instructions or human inputs.*

### Programa 3: Construir un Sistema RAG usando Pinecone para almacenamiento vectorial

1. **Configuración del Entorno y Dependencias**:
   - Importar los módulos y librerías necesarios.
   - Establecer las claves de la API de OpenAI y Pinecone como variables de entorno.

2. **Cargar Texto y Dividir Documentos**:
   - Cargar texto desde un archivo usando un cargador de texto.
   - Dividir los documentos en fragmentos más pequeños para su procesamiento.

3. **Crear y Gestionar el Índice de Pinecone**:
   - Inicializar un cliente de Pinecone.
   - Verificar si el índice deseado existe; si no, crearlo.

4. **Indexar Documentos**:
   - Utilizar PineconeVectorStore para crear un índice de representaciones vectoriales de los documentos.

5. **Realizar una Búsqueda**:
   - Inicializar PineconeVectorStore con un índice existente.
   - Realizar una búsqueda de similitud con una consulta.
   - Imprimir el contenido del documento más relevante.

#### Código

```Python
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec
import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "7e0a766f-a9d3-4d9f-b1f0-5f1538d57094"
os.environ["PINECONE_ENV"] = "gcp-starter"

def loadText():
    loader = TextLoader("Conocimiento.txt")
    documents = loader.load()
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
        is_separator_regex = False,
    )


    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    import pinecone


    index_name = "langchain-demo"
    pc = Pinecone(api_key='7e0a766f-a9d3-4d9f-b1f0-5f1538d57094')

    print(pc.list_indexes())

    # First, check if our index already exists. If it doesn't, we create it
    if len(pc.list_indexes())==0:
        # we create a new index
        #pc.create_index(name=index_name, metric="cosine", dimension=1536)
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(
                environment=os.getenv("PINECONE_ENV"),
                pod_type="p1.x1",
                pods=1
            )
        )

    # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
    docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

def search():
    embeddings = OpenAIEmbeddings()

    index_name = "langchain-demo"
    # if you already have an index, you can load it like this
    docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

    query = "What is the leading cause of avoidable premature deaths according to the European Commission, as mentioned in the text?"
    docs = docsearch.similarity_search(query)

    print(docs[0].page_content)

loadText()
search()

```

#### Resultado

![](/img/p3.png)

**Pregunta:** *What is the leading cause of avoidable premature deaths according to the European Commission, as mentioned in the text?*

**Salida:** *The world population is aging. In public health, there are several challenges resulting from this demographic change. One major issue concerns the expected increase in the prevalence of many chronic health conditions, which reduce the quality of life (World Health Organization, 2006). This is particularly concerning for non-communicable diseases (NCDs), which are currently “responsible for 80% of the disease burden in the EU countries and the leading causes of avoidable premature deaths” (European Commission, n.d.) Neurological disorders are NCDs that affect the central and peripheral nervous system, namely the brain, spinal cord, nerves, and muscles (World Health Organization, 2006). Some examples include dementias such as Alzheimer's disease, epilepsy, stroke, and other cerebrovascular diseases, migraine, multiple sclerosis, Parkinson’s disease, brain tumors, and traumatic disorders, among others (World Health Organization, 2016).*
## Análisis de Resultados
### Programa 1: Integración de ChatGPT con Lang Chain y OpenAI

### Programa 1: Integración de ChatGPT con Lang Chain y OpenAI
- **Desarrollo:** Este programa se enfoca en la interacción entre ChatGPT y Lang Chain/OpenAI para responder preguntas específicas.
- **Resultado:** La respuesta generada muestra una explicación detallada sobre el núcleo de la teoría de Popper, demostrando la capacidad de la integración para generar respuestas contextuales y significativas.

### Programa 2: Desarrollo de un RAG con una base de datos vectorial en memoria
- **Desarrollo:** Este programa construye un sistema de respuesta asistida por recuperación (RAG) utilizando una base de datos vectorial en memoria y herramientas como Lang Chain, Pinecone y OpenAI.
- **Resultado:** La respuesta proporcionada sobre la descomposición de tareas demuestra una comprensión precisa y contextual de la pregunta, lo que indica una implementación exitosa del sistema RAG.

### Programa 3: Implementación de un RAG utilizando Pinecone
- **Desarrollo:** Este programa utiliza Pinecone junto con Lang Chain para construir un sistema RAG utilizando una base de datos vectorial.
- **Funcionamiento de Pinecone:** Pinecone es utilizado para crear y gestionar un índice de similitud vectorial a partir de los documentos proporcionados. Esto implica la indexación de representaciones vectoriales de los fragmentos de texto de los documentos, lo que permite una búsqueda eficiente de similitud.
- **Resultado:** La respuesta obtenida sobre la principal causa de muertes prematuras evitables según la Comisión Europea, extraída del texto, demuestra una capacidad efectiva para recuperar información relevante de los documentos indexados en la base de datos vectorial.


## Conclusiones

Estas implementaciones ofrecen una visión amplia de las capacidades y aplicaciones del procesamiento de lenguaje natural (NLP) en Python. Desde el uso de modelos de lenguaje como ChatGPT para generar respuestas automáticas hasta la construcción de sistemas de recuperación de información eficientes utilizando bases de datos vectoriales en memoria o servicios en la nube como Pinecone, estas herramientas destacan la versatilidad y flexibilidad de las soluciones de NLP. Además, la integración de API como la de OpenAI simplifica enormemente el acceso a potentes modelos de lenguaje, permitiendo a los desarrolladores implementar sistemas sofisticados con relativa facilidad. Estas implementaciones demuestran cómo la combinación de diversas herramientas y técnicas en el campo del NLP puede conducir al desarrollo de sistemas poderosos y versátiles que aborden una amplia gama de problemas en el procesamiento de lenguaje natural.
