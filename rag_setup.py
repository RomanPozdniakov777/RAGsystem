import logging
import os
import chromadb
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Настройка логирования для отслеживания работы системы
logging.basicConfig(
    filename = 'rag_debug.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    encoding = 'utf-8',
)

class DocumentProcessor:
    """Класс для загрузки и обработки PDF-документов из указанной папки."""

    def __init__(self, folder = './documents'):
        '''
        Инициализирует процесс документов.
        Создает папку для документов, если она не существует.
        '''
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        logging.info(f'Инициализация DocumentProcessor с папкой: {folder}')

    def load_pdf_document(self, path):
        '''
        Загружает текст из одного PDF-файла.

        Args:
            path (str): Путь к PDF-файлу.

        Returns:
            str: Текст, извлеченный из PDF.
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
        Загружает все PDF-файлы из папки self.folder.

        Returns:
            list: Список словарей с ключами 'text' (содержимое) и 'source' (имя файла).
        '''

        logging.info(f'Поиск документов в папке: {self.folder}')

        documents = []

        for filename in os.listdir(self.folder):
            if filename.endswith('.pdf'):
                path = os.path.join(self.folder, filename)
                text = self.load_pdf_document(path)

                documents.append({
                    'text': text,
                    'source': filename,
                })

        logging.info(f'Загружено: {len(documents)} документов')
        return documents

class TextChunker:
    """Класс для разбиения текста на фрагменты (чанки) с перекрытием."""
    def __init__(self, chunk_size = 500, overlap = 80):
        '''
        Инициализирует чанкер.

        Args:
            chunk_size (int): Максимальный размер одного чанка в символах.
            overlap (int): Размер перекрытия между соседними чанками.
        """
        '''
        self.chunk_size = chunk_size
        self.overlap = overlap
        logging.info(f'Инициализация TextChunker: размер={chunk_size}, пересечение={overlap}')

    def split_text_into_chunks(self, text):
        '''
        Разбивает текст на чанки с заданным перекрытием.
        Разбиение происходит по предложениям (разделитель '. ').

        Args:
            text (str): Входной текст для разбиения.

        Returns:
            list: Список строк-чанков.
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
                # Сохраняем текущий чанк
                if current_chunk:
                   chunks.append('. '.join(current_chunk) + '.')

                # Создаем перекрытие: берем последние предложения из предыдущего чанка
                overlap_chunk = []
                overlap_length = 0
                for prev_sentence in reversed(current_chunk):
                    if overlap_length + len(prev_sentence) <= self.overlap:
                        overlap_chunk.insert(0, prev_sentence)
                        overlap_length += len(prev_sentence)
                    else:
                        break

                # Новый чанк = перекрытие + текущее предложение
                current_chunk = overlap_chunk + [sentence]
                current_length = overlap_length + sentence_length

        # Сохраняем последний чанк
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        logging.info(f'Создано чанков с пересечениями: {len(chunks)}')
        return chunks

class EmbeddingManager:
    """Класс для создания векторных представлений (эмбеддингов) текста."""

    def __init__(self):
        ''' Инициализирует менеджер эмбеддингов. '''
        self.model = None
        logging.info('Инициализация EmbeddingManager')

    def initialize_model(self):
        '''
        Загружает предобученную модель для создания эмбеддингов.
        Используется multilingual модель, поддерживающая русский и английский языки.
        '''
        logging.info('Загрузка модели для эмбеддингов')

        # Загрузка модели SentenceTransformer для создания эмбеддингов
        # Эта модель поддерживает множество языков, включая русский и английский
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        logging.info('Модель успешно загружена')

    def create_embeddings_for_chunks(self, chunks):
        '''
        Создает эмбеддинги для списка текстовых чанков.

        Args:
            chunks (list): Список текстовых чанков.

        Returns:
            numpy.ndarray: Массив эмбеддингов.

        Raises:
            ValueError: Если модель не была инициализирована.
        '''
        if self.model is None:
            raise ValueError('Модель не инициализирована. Вызовите initialize_model()')

        logging.info(f'Создание эмбеддингов для {len(chunks)} чанков')

        # Модель.encode автоматически обрабатывает список текстов
        embeddings = self.model.encode(chunks)
        return embeddings

class VectorDB:
    """Класс для взаимодействия с векторной базой данных ChromaDB."""
    def __init__(self, persist_directory = 'chroma_db'):
        '''
        Инициализирует клиент ChromaDB.

        Args:
            persist_directory (str): Путь к папке для хранения данных ChromaDB.
        '''
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        logging.info(f'Инициализация VectorDB в папке: {persist_directory}')

    def initialize_client(self):
        '''Инициализирует клиент ChromaDB и получает/создает коллекцию 'documents'.'''
        logging.info(f'Инициализация ChromaDB в папке: {self.persist_directory}')

        # Создаем папку для базы данных
        os.makedirs(self.persist_directory, exist_ok = True)

        # Используем способ инициализации PersistentClient
        self.client = chromadb.PersistentClient(path = self.persist_directory)

        # Получаем или создаем коллекцию с именем 'documents'
        self.collection = self.client.get_or_create_collection(name = 'documents')

        logging.info('ChromaDB успешно инициализирована')

    def save_chunks(self, chunks, embeddings, filename):
        '''
        Сохраняет чанки и их эмбеддинги в ChromaDB.

        Args:
            chunks (list): Список текстовых чанков.
            embeddings (numpy.ndarray): Массив эмбеддингов для чанков.
            filename (str): Имя исходного файла.
        '''
        if self.collection is None:
            raise ValueError('Хранилище не инициализировано. Вызовите initialize()')

        logging.info(f'Сохранение {len(chunks)} чанков в ChromaDB из файла: {filename}')

        # Создаем уникальные ID для каждого чанка
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

        # Создаем метаданные для каждого чанка
        metadatas = [
            {"source": filename, "chunk_id": i} for i in range(len(chunks))
        ]

        # Добавляем данные в коллекцию
        self.collection.add(
            embeddings = embeddings.tolist(),
            documents = chunks,
            metadatas = metadatas,
            ids = ids
        )
        logging.info(f'Успешно сохранено {len(chunks)} чанков')


class RAGOrchestrator:
    """Основной класс для координации процесса настройки RAG-системы."""
    def __init__(self):
        '''Инициализирует все компоненты RAG-системы.'''
        # Создаем экземпляры всех необходимых компонентов
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker()
        self.embedding_manager = EmbeddingManager()
        self.vector_db = VectorDB()

        logging.info('Инициализация RAGOrchestrator')

    def setup_rag_system(self):
        '''
        Выполняет полную настройку RAG-системы:
        1. Инициализирует модель для эмбеддингов
        2. Инициализирует векторную базу данных
        3. Загружает все документы из папки
        4. Обрабатывает каждый документ (чанкинг -> эмбеддинги -> сохранение)
        '''
        logging.info('Начало полной настройки RAG системы')

        try:
            # 1. Загружаем модель для создания эмбеддингов
            self.embedding_manager.initialize_model()

            # 2. Инициализируем векторную базу данных
            self.vector_db.initialize_client()

            # 3. Загружаем все документы из папки
            documents = self.document_processor.load_all_documents()

            # 4. Обрабатываем каждый документ по отдельности
            for document in documents:
                self._process_single_document(document)
            logging.info('Полная настройка RAG системы завершена успешно')
        except Exception as e:
            logging.error(f'Ошибка при настройке RAG системы: {str(e)}')
            raise

    def _process_single_document(self, document):
        '''
        Обрабатывает один документ: разбивает на чанки, создает эмбеддинги, сохраняет в БД.

        Args:
            document (dict): Словарь с ключами 'text' и 'source'.
        '''
        source = document['source']
        text = document['text']
        logging.info(f'Обработка документа: {source}')
        try:
            # 1. Разбиваем текст на чанки
            chunks = self.text_chunker.split_text_into_chunks(text)

            # 2. Создаем эмбеддинги для чанков
            embeddings = self.embedding_manager.create_embeddings_for_chunks(chunks)

            # 3. Сохраняем чанки и эмбеддинги в векторную базу
            self.vector_db.save_chunks(chunks, embeddings, source)

            logging.info(f'Документ {source} обработан успешно')

        except Exception as e:
            logging.error(f'Ошибка при обработке документа {source}: {str(e)}')
            raise