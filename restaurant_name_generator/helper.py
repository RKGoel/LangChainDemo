from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7)


def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for it."
    )

    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some food menu items for {restaurant_name}. \
        Return it as a comma separated list without index numbers."
    )

    restaurant_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
    items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")
    chain = SequentialChain(
        chains=[restaurant_chain, items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )

    items_from_seq_chain = chain({'cuisine': cuisine})

    return items_from_seq_chain


if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))
