# **Sistema de respuestas a grandes documentos**

### Descripción
Este proyecto implementa una solución que básicamente al entregar cualquier PDF, DOCX o Wikipedia a nuestro modelo y en base a este gran documento con la tecnologia de gpt-3.5-turbo
divide los textos y los almacena en embeddings, de tal forma que podemos preguntar cualquier cosa dentro del contexto del documento y este nos brindará una respuesta. Utiliza tecnologías
como Python, Pinecone, OpenAI y otras librerias para lograr sus objetivos.

![image](https://github.com/user-attachments/assets/4f56934c-6a15-4cf2-81c1-60abf833fe1d)

---

## **Estructura del proyecto**

- **`bigDocumentsResponses.ipynb`**: Archivo principal que ejecuta el flujo completo del programa.
- **`requirements.txt`**: Lista de dependencias necesarias para ejecutar el proyecto.

---

## **Instalación**

1. Clona este repositorio:
   ```bash
   git clone https://github.com/Jose47Morales/Python-Langchain.git
   ```
2. Ve al directorio del proyecto:
   ```bash
   cd ruta_descarga
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Configura las variables de entorno necesarias (PINECONE_API_KEY, OPENAI_API_KEY) en un archivo .env
   ```bash
   PINECONE_API_KEY='tu_api_key'
   OPENAI_API_KEY='tu_openai_key'
   ```

---

### **Uso**

## **Ejeución básica**

Abre el archivo principal (**`bigDocumentsResponses.ipynb`**) en Jupyter Notebook o Jupyter Lab:
1. Asegurate de tener Jupyter instalado:
   ```bash
   pip install notebook
   ```
2. Inicia Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Navega hasta el archivo **`bigDocumentsResponses.ipynb`** y ejecútalo paso a paso siguiendo las celdas.

---

### **Explicación del código**

## **1. Fragmentación de texto**

En el archivo principal, el primer paso del flujo es fragmentar un texto largo en partes más pequeñas para facilitar su análisis. Esto se realiza mediante la función **`fragmentar`**:

```bash
def fragmentar(data, chunk_size=150, chunk_overlap=20):
    # Dividimos el texto en fragmentos más pequeños para el procesamiento
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)
```

Esta función divide el texto en fragmentos manejables y devuelve una lista de objetos con atributos como el contenido del fragmento y su metadata.

## **2. Creación de índices en Pinecone**

El siguiente paso es almacenar los fragmentos como vectores en Pinecoe, lo que permite búsquedas semánticas rápidas. esto se realiza con la función **`creando_vectores`**:

```bash
def creando_vectores(fragmentos, index_name):
    import time
    from pinecone import Pinecone, ServerlessSpec
    from langchain.schema import Document
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

    pc = Pinecone(api_key = os.environ.get('PINECONE_API_KEY'))
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    from langchain.vectorstores import Pinecone
    if index_name not in existing_indexes:
        print(f'Creando el indice {index_name} y los embeddings ...', end = '')
        pc.create_index(name = index_name,
                        dimension = 1536, 
                        metric = 'cosine',
                        spec = ServerlessSpec(cloud = "aws", region = "us-east-1"),
                        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        vectores = Pinecone.from_documents(fragmentos, embeddings, index_name = index_name)
        print('Ok')
    else:
        print(f'El indice {index_name} ya existe. Cargando los embeddings ...', end = '')
        vectores = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')

    return vectores
```

Esta función verifica si el índice ya existe y, si no, lo crea. Luego, almacena los vectores generados a partir de los fragmentos.

## **Búsqueda semántica**

En el notebook también puedes realizar búsquedas en Pinecone para encontrar información relevante basada en similitud semántica:

```bash
def consulta_con_memoria(vectores, pregunta, memoria=[]):
    # Realizamos consultas con memoria de conversaciones previas.
    llm = ChatOpenAI(temperature=1)
    retriever = vectores.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    respuesta = crc({'question': pregunta, 'chat_history': memoria})
    memoria.append((pregunta, respuesta['answer']))
    return respuesta, memoria
```

Esto permite recibir una respuesta recuperando fragmentos relevantes basados en una consulta natural.

---

### **Tecnologías utilizadas**

* Python 3.8+
* Pinecone: Para la indexación y búsqueda vectorial.
* OpenAI: Para generar embeddings y modelos de lenguaje.
* Jupyter Notebook: Para análisis y experimentación

---

### **Contribuciones**

Las contribuciones son bienvenidas. Si deseas colaborar:

1. Haz un folk del repositorio
2. Crea una rama nueva:
   ```bash
   git checkout -b feature/nueva_funcionalidad
   ```
3. Realiza tus cambios y envía un pull request.

---

(c) Todos los derechos reservados. Creado por Jose47Morales y Mrmeepseeks. Si tienes preguntas, no dudes en contactar.
