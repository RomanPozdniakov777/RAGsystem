import logging
from rag_setup import EmbeddingManager

# Настройка логирования для модуля поиска
logging.basicConfig(
    filename = 'rag_debug.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    encoding = 'utf-8',
)

class Retriever:
    """Класс для поиска релевантных тектовых фрагментов (чанков) в векторной базе данных."""
    def __init__(self, vector_db = None):
        '''
        Инициализирует компонент поиска (retriever).

        Args:
            vector_db: Экземпляр VectorDB для выполнения поиска.
                       Если None, нужно будет передать его позже.
        '''
        self.vector_db = vector_db
        # Используем тот же EmbeddingManager, что и для индексации, чтобы создавать эмбеддинги запросов в том же пространстве
        self.embedding_manager = EmbeddingManager()
        logging.info('Инициализирован retriever')

    def initialize_retriever(self):
        '''
        Инициализирует модель для создания эмбеддингов запросов.
        Используется та же модель, что и для индексации документов.
        '''
        # Загружаем модель для создания эмбеддингов текста запроса пользователя
        self.embedding_manager.initialize_model()
        logging.info('Retriever модель инициализирована успешно')

    def search_relevant_chunks(self, query, n_results = 3):
        '''
        Ищет в векторной базе данных чанки, наиболее релевантные текстовому запросу.

        Args:
            query (str): Текстовый запрос пользователя.
            n_results (int): Количество релевантных чанков для возврата.

        Returns:
            list: Список текстов релевантных чанков.

        Raises:
            ValueError: Если векторная база или модель не инициализированы.
            Exception: При ошибке поиска.
        '''
        # Проверка, что векторная база данных инициализирована
        if self.vector_db is None or self.vector_db.collection is None:
            raise ValueError('Векторная база не инициализирована')

        # Проверка, что модель для создания эмбеддингов инициализирована
        if self.embedding_manager.model is None:
            raise ValueError('Модель retriever не инициализирована')

        logging.info(f'Поиск релевантных чанков для запроса: {query}')

        try:
            # 1. Создаем эмбеддинг для текстового запроса пользователя
            # create_embeddings_for_chunks ожидает список, поэтому оборачиваем query в [query]
            query_embedding = self.embedding_manager.create_embeddings_for_chunks([query])

            # 2. Выполняем поиск в векторной базе данных
            # query_embeddings должен быть списком списков (или numpy массивом)
            # query_embedding[0] - это первый (и единственный) эмбеддинг из списка
            # .tolist() преобразует numpy массив в обычный Python список
            results = self.vector_db.collection.query(
                query_embeddings = [query_embedding[0].tolist()],
                n_results = n_results
            )

            # 3. Извлекаем тексты найденных релевантных чанков из результатов
            # ChromaDB возвращает вложенные списки, поэтому берем [0]
            # чтобы получить плоский список текстов чанков
            relevant_chunks = results['documents'][0]
            logging.info(f'Найдено {len(relevant_chunks)} релевантных чанков')
            return relevant_chunks

        except Exception as e:
            logging.error(f'Ошибка при поиске релевантных чанков: {str(e)}')
            raise


