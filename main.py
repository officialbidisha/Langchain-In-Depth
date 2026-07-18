from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()


def main():

    print("Hello from langchain!")
    messages = [
        SystemMessage(content="You are a helpful assistant that translates corporate jargon into plain English."),
        HumanMessage(content="What is a good name for a company that makes SaaS Products?")
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, max_tokens=100)
    response = llm.invoke(messages)
    print(response.content)

    ## Multi turn conversation

    messages = [
        SystemMessage(content="You are a helpful assistant that translates corporate jargon into plain English."),
        HumanMessage(content="What is a good name for a company that makes Streaming Products?"),
    ]
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    messages.append(HumanMessage(content="Can you give me more options?"))
    response2 = llm.invoke(messages)
    print(response2.content)

    ### Runnable interface
    prompt2 = ChatPromptTemplate.from_template("Tell me a fact about {topic}.")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, max_tokens=100)
    parser = StrOutputParser()
    chain =prompt2|model|parser
    response3=chain.invoke({"topic":"the moon"})
    print(response3)
    









if __name__ == "__main__":
    main()
