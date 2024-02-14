import logging
from typing import Dict

# from json import JSONDecodeError

from operator import itemgetter
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.chains.base import Chain

from .prompts import (
    HATESPEECH_DETECTOR_PROMPT,
    HATESPEECH_VALIDATOR_PROMPT,
    HATESPEECH_CLASSIFICATION_PROMPT,
    RIGHT_WING_RATING_PROMPT,
)
from .prompts import (
    HatespeechDetectionFormat,
    HatespeechValidatorFormat,
    HatespeechClassifierFormat,
    RightWingRatingFormat,
)
from .llm_chains import get_openai_moderator_results
from .llm_chains import OpenAIModerationChain


_text_analyser_prompts = [
    (HATESPEECH_DETECTOR_PROMPT, HatespeechDetectionFormat),
    (HATESPEECH_VALIDATOR_PROMPT, HatespeechValidatorFormat),
    (HATESPEECH_CLASSIFICATION_PROMPT, HatespeechClassifierFormat),
    (RIGHT_WING_RATING_PROMPT, RightWingRatingFormat),
]


# TODO: add retry or output-fixing parser
def parallel_text_analyser_chains(
    llm: BaseChatModel = ChatOpenAI(),
    moderator: Chain = OpenAIModerationChain()
) -> RunnableParallel:
    """Parallel running chain, all sharing the same LLM. This chain is
    restricted to the prompts listed in _text_analyser_prompts. The parallel
    chain invokes 4 times the llm with 4 different prompts. Additionaly it
    passes the input message to a moderation chain. To invoke the you need to
    pass a dictionary with "message" as a key.

    Args:
        llm (BaseChatModel, optional):
            Pass a llm instance which works with langchain's PromptTemplate
            and JSON Parser. Defaults to ChatOpenAI().
        moderator (Chain, optional):
            Pass an additional chain to get the result of the moderator.
            Defaults to OpenAIModerationChain(). (from llm_chains)

    Returns:
        RunnableParallel: Runnable chain which you can invoke (and other
        methods)

    Example:
        from ki_gegen_rechts.analyser import parallel_text_analyser_chains
        llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.0)
        analyser_full_chain = parallel_text_analyser_chains(llm)
        result = analyser_full_chain.invoke({"message": "You suck!"})
    """
    chains = {}
    for prompt, parser_object in _text_analyser_prompts:
        parser = JsonOutputParser(pydantic_object=parser_object)
        chains[prompt.name] = prompt | llm | parser

    # Setting up detector with validator chain
    detector_chain = chains["detector"] | {
        "message": RunnablePassthrough(),
        "classification": itemgetter("classification"),
        "explanation": itemgetter("explanation"),
    }
    # TODO: How can I concat both chains but getting the results of both chains
    # at the end? The 2nd chain needs the input of the first chain. hacky workaround atm
    validator_chain = {
        "detector": {
            "classification": itemgetter("classification"),
            "explanation": itemgetter("explanation"),
        },
        "validator": chains["validator"],
    }

    map_chain = RunnableParallel(
        detection=(detector_chain | validator_chain),
        classifier=chains["classifier"],
        right_wing_rater=chains["right-wing-rater"],
        moderator=moderator,
    )

    return map_chain


# This function takes 10-20 seconds per message, because it invokes the
# chains in a loop. First attempt -> probably can be deleted
def analyse_text_message_with_llm(
    message: str, llm: BaseChatModel = ChatOpenAI()
) -> Dict[str, str]:
    input, results = {"message": message}, {}
    results["moderator"] = get_openai_moderator_results(message)

    # Initialize LLM chains
    for prompt, parser_object in _text_analyser_prompts:
        parser = JsonOutputParser(pydantic_object=parser_object)
        chain = prompt | llm | parser
        try:
            chain_output: dict = chain.invoke(input)
        except OutputParserException:
            logging.warning("Parsing failed - Switching parser to StrOutputParser")
            chain = prompt | llm | StrOutputParser()
            chain_output: str = chain.invoke(input)

        if prompt.name == "detector":
            input.update(chain_output)

        results[prompt.name] = chain_output

    return results
