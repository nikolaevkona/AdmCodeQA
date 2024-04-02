"""Module for chat determining the amount of the administrative fine."""

from typing import Optional, Dict
from . import AdminData
from openai import OpenAI
import openai
import backoff

class ChatClient:
    """Chat-GPT client for administrative fine determining.

    Args:
        api_key (str): Open-AI api-key.
        database (AdminData): Vector database of code articles.
        top_k (int): Top-k articles for retrieval.
        qa_template (:obj:`str`, optional): Template for fine QA.
        code_template (:obj:`str`, optional): Template for code QA.
        generation_kwargs (:obj:`dict`, optional): Chat-gpt generation kwargs.
            Including model
    """

    def __init__(self, api_key: str, database: AdminData, top_k: int = 3,
                 qa_template: Optional[str] = None,
                 code_template: Optional[str] = None,
                 generation_kwargs: Optional[Dict] = None) -> None:
        self._openai_client = OpenAI(api_key=api_key)
        self._database = database
        self._top_k = top_k

        if qa_template is None:
            self._qa_template = "Контекст:\n{}\n" \
                "Используя контекст, коротко ответь, какой штраф" \
                "установлен за следующее правонарушение "\
                "(в качестве ответа дай только денежную сумму): {}"
        else:
            self._qa_template = qa_template

        if code_template is None:
            self._code_template = "Контекст:\n{}\n" \
                "Используя контекст, коротко ответь, какая статья " \
                "определяет штраф за следующее правонарушение " \
                "(в качестве ответа дай только номер статьи): {}\n"
        else:
            self._code_template = code_template

        if generation_kwargs is None:
            self._generation_kwargs = {"model": "gpt-3.5-turbo", "top_p": 0}
        else:
            self._generation_kwargs = generation_kwargs

    def respond(self, query: str) -> str:
        """Get respond for query

        Args:
            query (str): Query text.

        Returns:
            respond: Text of respond.
        """
        context = self._database.retrieve(query, self._top_k)
        fine_query = self._qa_template.format(context, query)
        fine_respond = self._get_api_respond(fine_query)
        code_query = self._code_template.format(context, query)
        code_respond = self._get_api_respond(code_query)

        return f"Ответ: {fine_respond}\nНорма: КоАП РФ {code_respond}"

    def _get_api_respond(self, content: str) -> str:
        messages = [{"role": "user", "content": content}]
        request = self._completions_with_backoff(messages=messages,
                                                 **self._generation_kwargs)
        return request.choices[0].message.content

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def _completions_with_backoff(self, **kwargs):
        return self._openai_client.chat.completions.create(**kwargs)
