import logging
import os

# Импортируем основные компоненты RAG-системы
from rag_setup import RAGOrchestrator
from rag_retriever import Retriever
from rag_generator import Generator
from rag_setup import VectorDB

# Настройка логирования для главного файла
logging.basicConfig(
    filename = 'rag_main.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    encoding = 'utf-8'
)

def setup_rag_system():
    '''
    Выполняет первоначальную настройку RAG-системы.
    Обрабатывает все PDF-документы из папки и сохраняет их в векторной базе.
    '''
    logging.info('=== НАЧАЛО НАСТРОЙКИ RAG СИСТЕМЫ ===')

    try:
        # Создаем оркестратор - главный координатор настройки
        orchestrator = RAGOrchestrator()

        # Запускаем полный процесс настройки системы
        orchestrator.setup_rag_system()
        logging.info('=== НАСТРОЙКА RAG СИСТЕМЫ ЗАВЕРШЕНА УСПЕШНО ===')

    except Exception as e:
        logging.error(f'Ошибка при настройке RAG системы: {str(e)}')
        raise

def run_rag_query(query):
    '''
    Обрабатывает один запрос пользователя.

    Args:
        query (str): Вопрос пользователя.

    Returns:
        str: Сгенерированный ответ.
    '''
    logging.info(f'=== ВЫПОЛНЕНИЕ ЗАПРОСА: {query} ===')

    try:
        # 1. Инициализируем векторную базу данных для поиска
        vector_db = VectorDB()
        vector_db.initialize_client()

        # 2. Создаем и инициализируем компонент поиска (retriever)
        retriever = Retriever(vector_db)
        retriever.initialize_retriever()

        # 3. Ищем релевантные фрагменты текста в базе данных
        relevant_chunks = retriever.search_relevant_chunks(query)

        # 4. Создаем и инициализируем генератор ответов
        generator = Generator()
        generator.initialize_generator()

        # 5. Генерируем ответ на основе найденных фрагментов
        answer = generator.generate_answer(query, relevant_chunks)

        logging.info('=== ЗАПРОС ВЫПОЛНЕН УСПЕШНО ===')

        return answer

    except Exception as e:
        logging.error(f'Ошибка при выполнении запроса: {str(e)}')
        raise

def main():
    '''
    Главная функция программы.
    Координирует процесс настройки и выполнения запросов.
    '''
    logging.info('=== ЗАПУСК RAG СИСТЕМЫ ===')

    try:
        # Проверяем, существует ли уже база данных
        # Если нет - выполняем первоначальную настройку
        if not os.path.exists('./chroma_db'):
            setup_rag_system()
        else:
            logging.info('База данных уже существует, пропускаем setup')

        # Основной цикл обработки запросов пользователя
        while True:
            query = input("\nВведите ваш вопрос (или 'выход' для завершения): ")

            # Проверяем команду выхода
            if query.lower() in ['выход', 'exit']:
                logging.info('Пользователь завершил работу')
                break

            # Обрабатываем непустой запрос
            if query.strip():
                answer = run_rag_query(query)
                print(f"\nОтвет: {answer}")
            else:
                print("Пожалуйста, введите непустой запрос")

        logging.info('=== RAG СИСТЕМА ЗАВЕРШИЛА РАБОТУ ===')

    except KeyboardInterrupt:
        # Обработка прерывания пользователем (Ctrl+C)
        logging.info('Работа прервана пользователем (Ctrl+C)')
        print("\nРабота завершена пользователем")

    except Exception as e:
        logging.error(f'Критическая ошибка в main: {str(e)}')
        print(f"Критическая ошибка: {str(e)}")

# Точка входа в программу
if __name__ == "__main__":
    main()