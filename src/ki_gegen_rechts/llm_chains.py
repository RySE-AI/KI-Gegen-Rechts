from typing import Any, Dict, List, Mapping, Optional, Union, Tuple

import os

from langchain.chains.base import Chain
from langchain_openai import ChatOpenAI
from langchain.prompts import Prompt
from langchain_core.output_parsers import BaseOutputParser
from openai import OpenAI, AsyncOpenAI
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_core.runnables.base import RunnableSequence
from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser


# TODO: Check if a inheritance from BaseOpenAI is "better"
# Combination of BaseOpenAI and OpenAIModerationChain from langchain
class OpenAIModerationChain(Chain):
    """Pass input through a moderation endpoint.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    This Chain returns the whole results of the moderation model - it does
    not directly work like the community version.

    Example:
        .. code-block:: python

            from langchain.chains import OpenAIModerationChain
            moderation = OpenAIModerationChain()
    """

    client: Any  #: :meta private:
    model_name: Optional[str] = "text-moderation-latest"
    """Moderation model name to use."""
    error: bool = False
    """Whether or not to error if bad content was found."""
    input_key: str = "message"  #: :meta private:
    output_key: str = "mod_results"  #: :meta private:
    openai_api_key: Optional[str] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `OPENAI_API_KEY` if not provided."""
    openai_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    openai_organization: Optional[str] = Field(default=None, alias="organization")
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    max_retries: int = 2
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = None
    """Optional httpx.Client."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""

        values["openai_api_key"] = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        values["openai_api_base"] = values["openai_api_base"] or os.getenv(
            "OPENAI_API_BASE"
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        values["openai_organization"] = (
            values["openai_organization"]
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )

        client_params = {
            "api_key": values["openai_api_key"],
            "organization": values["openai_organization"],
            "base_url": values["openai_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
            "http_client": values["http_client"],
        }
        if not values.get("client"):
            values["client"] = OpenAI(**client_params).moderations
        if not values.get("async_client"):
            values["async_client"] = AsyncOpenAI(**client_params).moderations

        return values

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        text = inputs[self.input_key]
        outputs = self.client.create(input=text, model=self.model_name)
        return {self.output_key: outputs.results[0].model_dump()}


def create_public_chat_gpt_chain(
    prompt: Prompt,
    parser: BaseOutputParser,
    model: str = "gpt-3.5-turbo-1106",
    model_kwargs: Dict = {},
) -> RunnableSequence:
    """Simple LCEL chain for OpenAI which is build with a prompt, a llm and
    a parser. The function returns a langchain chain.

    Args:
        prompt (Prompt):
            The initial input prompt that will be passed to the LLM.
        parser (BaseOutputParser):
            The parser that processes the output of the LLM.
        model (str, optional):
            The identifier of the OpenAI model to be used. Defaults to
            "gpt-3.5-turbo-1106".
        model_kwargs (dict, optional):
            Additional keyword arguments to be passed to the LLM instantiation.
            Defaults to {}.

    Returns:
        RunnableSequence: LCEL Chain
    """
    openai_llm = ChatOpenAI(model=model, temperature=0.0, **model_kwargs)
    chain = prompt | openai_llm | parser

    return chain


def get_openai_moderator_results(
    input_message: str, model: str = "text-moderation-latest"
) -> Dict:
    """Get the moderation result of the passed input message

    Args:
        input_message (str):
            Message you want to check with openai's moderation model
        model (str, optional):
            moderation model -> look at openai's documentation.
            Defaults to "text-moderation-latest".

    Returns:
        Dict: _description_
    """
    client = OpenAI()
    output = client.moderations.create(input=input_message, model=model)
    results = output.results[0].model_dump()

    return results


# TODO: NOT TESTED
class JsonBooleanChecker(BaseCumulativeTransformOutputParser[Any]):
    """Check the output of a JSON to check if boolean are parsed correctly"""

    true_vals = ["true", "right", "yes"]
    false_vals = ["false", "wrong", "no"]

    def parse_result(self, result: List[dict]) -> Any:
        for key, val in result.items():
            if not isinstance(val, bool) and val.lower() in self.true_vals:
                result[key] = True
            elif not isinstance(val, bool) and val.lower() in self.false_vals:
                result[key] = False
            else:
                continue

        return result

    def parse(self, result: dict) -> Any:
        return self.parse_result(result)

    @property
    def _type(self) -> str:
        return "json_output_boolean_parser"
