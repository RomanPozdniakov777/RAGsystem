import logging
import torch
from transformers import pipeline

logging.basicConfig(
    filename = 'rag_debug.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    encoding = 'utf-8',
)

class Generator:
    def __init__(self, model_name = 'IlyaGusev/siberia7b_gradio'):
        '''

        :param model_name:
        '''
        self.model_name = model_name
        self.generator = None
        logging.info(f'Generator инициализирован с моделью: {model_name}')

    def initialize_generator(self):
        '''

        :return:
        '''
        try:
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )

            logging.info('Генеративная модель инициализирована успешно')

        except Exception as e:
            logging.error(f'Ошибка при инициализации генератора: {str(e)}')
            raise

    def generate_answer(self, query, relevant_chunks):
        '''
        :param query:
        :param relevant_chunks:
        :return:
        '''
        if self.model is None:
            raise ValueError('Генеративная модель не инициализирована')

        logging.info(f'Генерация ответа для запроса: {query}')
        try:
            context = "\n".join(relevant_chunks)

            prompt = f"""
                    Используй следующую информацию для ответа на вопрос.
                    Если информации недостаточно, скажи об этом честно.

                    Контекст:
                    {context}

                    Вопрос: {query}

                    Ответ:
                    """

            answer = self.model(
                prompt,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )

            answer = answer.strip()

            logging.info('Ответ сгенерирован успешно')

            return answer

        except Exception as e:
            logging.error(f'Ошибка при генерации ответа: {str(e)}')
            raise