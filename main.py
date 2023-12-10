from dotenv import load_dotenv, find_dotenv

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    # print(os.getenv("OPENAI_API_KEY"))

    # temperature param tunes creativity of model
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.6)
    # response = llm("explain large language model in one sentence")

    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for it."
    )

    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some food menu items for {restaurant_name}. Return it as a comma separated list."
    )

    ## Execute only one prompt with chain
    restaurant_chain = LLMChain(llm=llm, prompt=prompt_template_name)
    # restaurant_name = restaurant_chain.run("Indian")
    # print(restaurant_name)

    items_chain = LLMChain(llm=llm, prompt=prompt_template_items)
    # items_name = items_chain.run(restaurant_name)
    # print(items_name)

    ## Execute prompts in simple sequential chain
    # chain = SimpleSequentialChain(chains=[restaurant_chain, items_chain], verbose=True)
    # items_from_simple_seq_chain = chain.run("Indian")
    # print(items_from_simple_seq_chain)

    # Execute prompts using sequential chain
    restaurant_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
    items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="items_name")
    chain = SequentialChain(
        chains=[restaurant_chain, items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'items_name']
    )

    items_from_seq_chain = chain({'cuisine':'Mexican'})
    print(items_from_seq_chain)
