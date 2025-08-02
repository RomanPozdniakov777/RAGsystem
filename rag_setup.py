import logging
import os
import chromadb
from PyPDF2 import PdfReader, PdfFileReader
from chromadb import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    filename = 'rag_debug.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    encoding = 'utf-8',
)

class DocumentProcessor:
    def __init__(self, folder = './documents'):
        '''

        '''
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        logging.info(f'Инициализация DocumentProcessor с папкой: {folder}')

    def load_pdf_document(self, path):
        '''

        :param path:
        :return:
        '''

        logging.info(f'Чтение файла: {path}')
        reader = PdfReader(path)
        text = ''

        for page in reader.pages:
            text += page.extract_text()

        logging.info(f'Загружено страниц: {len(reader.pages)}')
        return text

    def load_all_documents(self):
        '''

        :param path:
        :return:
        '''

        logging.info(f'Поиск документов в папке: {self.folder}')

        documents = []

        for filename in os.listdir(self.folder):
            if filename.endswith('.pdf'):
                path = os.path.join(path, filename)
                text = self.load_pdf_document(path)

                documents.append({
                    'text': text,
                    'source': filename,
                })

        logging.info(f'Загружено: {len(documents)} документов')
        return documents

class TextChunker:
    def __init__(self, chunk_size = 500, overlap = 80):
        '''

        '''
        self.chunk_size = chunk_size
        self.overlap = overlap
        logging.info(f'Инициализация TextChunker: размер={chunk_size}, пересечение={overlap}')

    def split_text_into_chunks(self, text):
        '''

        :param text:
        :param chunk_size:
        :param overlap:
        :return:
        '''
        logging.info(f'Разбиение текста на чанки: размер {self.chunk_size}, пересечение {self.overlap}')

        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                   chunks.append('. '.join(current_chunk) + '.')

                overlap_chunk = []
                overlap_length = 0
                for prev_sentence in reversed(current_chunk):
                    if overlap_length + len(prev_sentence) <= self.overlap:
                        overlap_chunk.insert(0, prev_sentence)
                        overlap_length += len(prev_sentence)
                    else:
                        break

                current_chunk = overlap_chunk + [sentence]
                current_length = overlap_length + sentence_length

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        logging.info(f'Создано чанков с пересечениями: {len(chunks)}')
        return chunks

class EmbeddingManager:
    def __init__(self):
        '''

        '''
        self.model = None
        logging.info('Инициализация EmbeddingManager')

    def initialize_model(self):
        '''

        :return:
        '''
        logging.info('Загрузка модели для эмбеддингов')

        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        logging.info('Модель успешно загружена')

    def create_embeddings_for_chunks(self, chunks):
        '''

        :param chunks:
        :param model:
        :return:
        '''
        if self.model is None:
            raise ValueError('Модель не инициализирована. Вызовите initialize_model()')

        logging.info(f'Создание эмбеддингов для {len(chunks)} чанков')

        embeddings = self.model.encode(chunks)
        return embeddings

class VectorDB:
    def __init__(self, persist_directory = './chroma_db'):
        '''

        :param persist_directory:
        '''
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        logging.info(f'Инициализация VectorDB в папке: {persist_directory}')

    def initialize_client(self):
        '''

        :return:
        '''
        logging.info(f'Инициализация ChromaDB в папке: {self.persist_directory}')

        # settings = Settings(
        #     chroma_db_impl = "duckdb+parquet",
        #     persist_directory = self.persist_directory,
        # )
        #
        # self.client = chromadb.Client(settings)
        self.client = chromadb.PersistentClient(path = self.persist_directory)
        self.collection = self.client.get_or_create_collection(name = 'documents')

        logging.info('ChromaDB успешно инициализирована')

    def save_chunks(self, chunks, embeddings, filename):
        '''

        :param chunks:
        :param embeddings:
        :param filename:
        :return:
        '''
        if self.collection is None:
            raise ValueError('Хранилище не инициализировано. Вызовите initialize()')

        logging.info(f'Сохранение {len(chunks)} чанков в ChromaDB из файла: {filename}')

        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

        metadatas = [
            {"source": filename, "chunk_id": i} for i in range(len(chunks))
        ]

        self.collection.add(
            embeddings = embeddings.tolist(),
            documents = chunks,
            metadatas = metadatas,
            ids = ids
        )
        logging.info(f'Успешно сохранено {len(chunks)} чанков')


class RAGOrchestrator:
    def __init__(self):
        '''

        '''
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker()
        self.embedding_manager = EmbeddingManager()
        self.vector_db = VectorDB()

        logging.info('Инициализация RAGOrchestrator')

    def setup_rag_system(self):
        '''

        :return:
        '''
        logging.info('Начало полной настройки RAG системы')

        try:
            self.embedding_manager.initialize_model()
            self.vector_db.initialize_client()
            documents = self.document_processor.load_all_documents()
            for document in documents:
                self._process_single_document(document)
            logging.info('Полная настройка RAG системы завершена успешно')
        except Exception as e:
            logging.error(f'Ошибка при настройке RAG системы: {str(e)}')
            raise

    def _process_single_document(self, document):
        '''

        :param document:
        :return:
        '''
        source = document['source']
        text = document['text']
        logging.info(f'Обработка документа: {source}')
        try:
            chunks = self.text_chunker.split_text_into_chunks(text)

            embeddings = self.embedding_manager.create_embeddings_for_chunks(chunks)

            self.vector_db.save_chunks(chunks, embeddings, source)

            logging.info(f'Документ {source} обработан успешно')

        except Exception as e:
            logging.error(f'Ошибка при обработке документа {source}: {str(e)}')
            raise