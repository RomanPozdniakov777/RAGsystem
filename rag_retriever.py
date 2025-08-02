import logging
from rag_setup import EmbeddingManager

logging.basicConfig(
    filename = 'rag_debug.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    encoding = 'utf-8',
)

class Retriever:
    def __init__(self, vector_db = None):
        '''
        :param vector_db:
        '''
        self.vector_db = vector_db
        self.embedding_manager = EmbeddingManager()
        logging.info('Инициализирован retriever')

    def initialize_retriever(self):
        '''
        :return:
        '''
        self.embedding_manager.initialize_model()
        logging.info('Retriever модель инициализирована успешно')

    def search_relevant_chunks(self, query, n_results = 3):
        '''
        :param query:
        :param n_results:
        :return:
        '''
        if self.vector_db is None or self.vector_db.collection is None:
            raise ValueError('Векторная база не инициализирована')

        if self.embedding_manager.model is None:
            raise ValueError('Модель retriever не инициализирована')

        logging.info(f'Поиск релевантных чанков для запроса: {query}')

        try:
            query_embedding = self.embedding_manager.create_embeddings_for_chunks([query])

            results = self.vector_db.collection.query(
                query_emeddings = [query_embedding[0].to_list()],
                n_results = n_results
            )

            relevant_chunks = results['documents'][0]
            logging.info(f'Найдено {len(relevant_chunks)} релевантных чанков')
            return relevant_chunks

        except Exception as e:
            logging.error(f'Ошибка при поиске релевантных чанков: {str(e)}')
            raise


