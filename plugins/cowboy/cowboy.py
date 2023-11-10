# encoding:utf-8

import plugins
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from channel.chat_message import ChatMessage
from common.log import logger
from plugins import *
from config import conf

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


@plugins.register(
    name="Cowboy",
    desire_priority=-1,
    hidden=True,
    desc="A cowboy plugin",
    version="0.1",
    author="liangdong",
)
class Cowboy(Plugin):
    def __init__(self):
        super().__init__()
        self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
        logger.info("[Cowboy] inited")


    def url_get(self, query):
        url = query
        prompt_template = """Write a concise summary of the following:
        {text}
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)

        refine_template = (
            "Your job is to produce a final summary\n"
            "We have provided an existing summary up to a certain point: {existing_answer}\n"
            "We have the opportunity to refine the existing summary"
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, refine the original summary in Chinese"
            "If the context isn't useful, return the original summary."
        )
        refine_prompt = PromptTemplate.from_template(refine_template)

        loader = WebBaseLoader(url)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=0, 
            separators=[" ", ",","，","。", "\n"]
        )

        split_docs = text_splitter.split_documents(docs)

        llm = ChatOpenAI(temperature=0, 
            model_name=conf().get("model", "gpt-3.5-turbo"),
            openai_api_base=conf().get("open_ai_api_base", "https://api.openai.com/v1"),
            openai_api_key=conf().get("open_ai_api_key", "")
        )
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )
        result = chain({"input_documents": split_docs}, return_only_outputs=True)
        output = result["output_text"]
        logger.info("[cowboy] url_get output: %s" % output)
        return output

    def on_handle_context(self, e_context: EventContext):
        logger.debug("[cowboy] on_handle_context accpeted, e_context: %s" % e_context["context"])
        if e_context["context"].type not in [
            ContextType.TEXT,
        ]:
            return

        content = e_context["context"].content
        content_list = e_context["context"].content.split(maxsplit=1)

        if not content or len(content_list) < 1:
            e_context.action = EventAction.CONTINUE
            return


        trigger_prefix = conf().get("plugin_trigger_prefix", "$")
        if not content.startswith(f"{trigger_prefix}cowboy"):
            return

        logger.info("[cowboy] on_handle_context. content: %s" % content)

        query = content_list[1].strip()


        reply = Reply()
        reply.type = ReplyType.TEXT
        msg: ChatMessage = e_context["context"]["msg"]
        if query == "SenseCore":
            reply.content = f"Time to Make SenseCore! {msg.from_user_nickname}"
        elif query.startswith('http'):
            reply.content = self.url_get(query)
        else:
            reply.content = "努力施工中。。。"
        e_context["reply"] = reply
        e_context.action = EventAction.BREAK_PASS



    def get_help_text(self, **kwargs):
        help_text = "输入Hello，我会回复你的名字\n输入End，我会回复你世界的图片\n"
        return help_text
