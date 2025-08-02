import logging
import os
from rag_setup import RAGOrchestrator
from rag_retriever import Retriever
from rag_generator import Generator
from rag_setup import VectorDB

logging.basicConfig(
    filename = 'rag_main.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    encoding = 'utf-8'
)

def setup_rag_system():
    '''
    :return:
    '''
    logging.info('=== НАЧАЛО НАСТРОЙКИ RAG СИСТЕМЫ ===')

    try:
        orchestrator = RAGOrchestrator()
        orchestrator.setup_rag_system()
        logging.info('=== НАСТРОЙКА RAG СИСТЕМЫ ЗАВЕРШЕНА УСПЕШНО ===')

    except Exception as e:
        logging.error(f'Ошибка при настройке RAG системы: {str(e)}')
        raise

def run_rag_query(query):
    '''
    :param query:
    :return:
    '''
    logging.info(f'=== ВЫПОЛНЕНИЕ ЗАПРОСА: {query} ===')

    try:
        vector_db = VectorDB()
        vector_db.initialize_client()

        retriever = Retriever(vector_db)
        retriever.initialize_retriever()

        relevant_chunks = retriever.search_relevant_chunks(query)

        generator = Generator()
        generator.initialize_generator()

        answer = generator.generate_answer(query, relevant_chunks)

        logging.info('=== ЗАПРОС ВЫПОЛНЕН УСПЕШНО ===')

        return answer

    except Exception as e:
        logging.error(f'Ошибка при выполнении запроса: {str(e)}')
        raise

def main():
    '''
    :return:
    '''
    logging.info('=== ЗАПУСК RAG СИСТЕМЫ ===')

    try:
        if not os.path.exists('./chroma_db'):
            setup_rag_system()
        else:
            logging.info('База данных уже существует, пропускаем setup')

        while True:
            query = input("\nВведите ваш вопрос (или 'выход' для завершения): ")

            if query.lower() in ['выход', 'exit']:
                logging.info('Пользователь завершил работу')
                break

            if query.strip():
                answer = run_rag_query(query)
                print(f"\nОтвет: {answer}")
            else:
                print("Пожалуйста, введите непустой запрос")

        logging.info('=== RAG СИСТЕМА ЗАВЕРШИЛА РАБОТУ ===')

    except KeyboardInterrupt:
        logging.info('Работа прервана пользователем (Ctrl+C)')
        print("\nРабота завершена пользователем")

    except Exception as e:
        logging.error(f'Критическая ошибка в main: {str(e)}')
        print(f"Критическая ошибка: {str(e)}")

if __name__ == "__main__":
    main()