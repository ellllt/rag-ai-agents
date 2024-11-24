from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableField

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
llm = llm.configurable_fields(max_tokens=ConfigurableField(id="max_tokens"))

