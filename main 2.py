from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()


def main():

    print("Hello from langchain!")
    prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    print(prompt)
    print (prompt.format(product="SaaS Products"))


if __name__ == "__main__":
    main()
