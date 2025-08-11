import logging
import torch
from transformers import pipeline

# Настройка логирования для отслеживания работы генератора
logging.basicConfig(
    filename = 'rag_debug.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    encoding = 'utf-8',
)

class Generator:
    """Класс для генерации ответов на вопросы пользователя на основе найденного контекста."""
    def __init__(self, model_name = "Qwen/Qwen3-0.6B"):
        '''
        Инициализирует генератор с указанной моделью.

        Args:
            model_name (str): Идентификатор модели на Hugging Face.
                              По умолчанию используется Qwen/Qwen3-0.6B.
        '''
        self.model_name = model_name
        self.generator = None
        logging.info(f'Generator инициализирован с моделью: {model_name}')

    def initialize_generator(self):
        '''
        Загружает и инициализирует генеративную модель через transformers.pipeline.
        Модель загружается в автоматически определяемое устройство (GPU/CPU).
        '''
        try:
            # Создание пайплайна для генерации текста
            self.generator = pipeline(
                "text-generation", # Тип задачи: генерация текста
                model = self.model_name, # Имя модели
                # Используем float16 для GPU (экономия памяти), float32 для CPU
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
                # Автоматический выбор устройства (GPU если доступен)
                device_map="auto" if torch.cuda.is_available() else None,
                # Необходимо для моделей с пользовательским кодом
                trust_remote_code = True
            )

            logging.info('Генеративная модель инициализирована успешно')

        except Exception as e:
            logging.error(f'Ошибка при инициализации генератора: {str(e)}')
            raise

    def generate_answer(self, query, relevant_chunks):
        '''
        Генерирует ответ на вопрос пользователя, используя найденный контекст.

        Args:
            query (str): Вопрос пользователя.
            relevant_chunks (list): Список релевантных текстовых фрагментов из БД.

        Returns:
            str: Сгенерированный ответ на русском языке.

        Raises:
            ValueError: Если модель не была инициализирована.
            Exception: При ошибке генерации.
        '''
        # Проверка, что модель инициализирована
        if self.generator is None:
            raise ValueError('Генеративная модель не инициализирована')

        logging.info(f'Генерация ответа для запроса: {query}')
        try:
            # Объединяем найденные релевантные фрагменты в один контекст
            context = "\n".join(relevant_chunks)

            # Формируем промпт для модели
            # Модель будет использовать этот текст как основу для генерации ответа
            prompt = f"Вопрос: {query}\nКонтекст: {context}\nОтвет на русском языке:"

            # Генерация ответа с заданными параметрами
            response = self.generator(
                prompt, # Входной текст (промпт)
                max_new_tokens = 150, # Максимальное количество генерируемых токенов
                temperature = 0.7, # Контроль креативности (0 - детерминировано, 1+ - креативно)
                top_p = 0.9, # Nucleus sampling - ограничивает выбор токенов по вероятности
                repetition_penalty = 1.2, # Штраф за повторение токенов
                do_sample = True # Использовать сэмплинг вместо жадного поиска
            )

            # Извлекаем только сгенерированную часть ответа
            # Убираем исходный промпт из результата
            full_text = response[0]['generated_text']
            answer = full_text[len(prompt):].strip()

            # Резервный вариант, если извлечение по длине не сработало
            if not answer:
                answer = full_text.strip()

            logging.info('Ответ сгенерирован успешно')

            # Возвращаем только сгенерированный ответ, без контекста
            return answer

        except Exception as e:
            logging.error(f'Ошибка при генерации ответа: {str(e)}')
            raise
