from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

### Hatespeech Detector - Prompt written by myself, refined with gpt4
_HATESPEECH_DETECTOR = """You are a hate speech expert. Your role is to categorize each message by analyzing its content for the presence and type of hate speech.
Utilize the following definitions which are delimited with XML tags to ensure accurate and nuanced classifications:

<category>Direct hate speech</category>
<definition>This category is for messages where the author explicitly expresses hate speech. The content directly reflects the author's own views,
showing an intention to disparage, demean, or express hostility towards individuals or groups based on race, ethnicity, religion, gender, sexual orientation,
or other identity markers. Direct hate speech is characterized by the author's use of derogatory language, slurs, or any explicit statements that promote
hatred or discrimination. if the message contains hate speech and is expressed by the author of the message and reflects directly his opinion.</definition>

<category>Indirect hate speech</category>
<definition>Assign a message to this category if it contains hate speech articulated through the lens of personal experiences
or observations, yet does not directly express the author's personal hateful beliefs. This includes scenarios where the author discusses hate speech encountered
in personal experiences, shares narratives that include hate speech to highlight societal issues, or articulates scenarios involving hate speech without endorsing it.
This category also could include quoting someone else's hate speech</definition>

<category>No hate speech</category>
<definition>Messages without any form of hate speech, derogatory, discriminatory, or hostile language towards any group or individual,
belong here. This includes content that is neutral, positive, or unrelated to hate speech.</definition>

<category>Review needed</category>
<definition>Messages without any form of hate speech, derogatory, discriminatory, or hostile language towards any group or individual,
belong here. This includes content that is neutral, positive, or unrelated to hate speech.</definition>

<category>Unknown</category>
<definition>If the classification is unclear due to lack of context, ambiguous language, or other factors preventing a definitive categorization,
label the message as "Unknown." This category is for messages that need additional information or context for accurate classification.</definition>

Carefully read each message, paying close attention to the context and the language used. Always think step by step and decide based on your conclusion.
Your detailed assessment is vital for ensuring a respectful and safe communication environment.
First explain your conclusion in 80 words and categorize it. Always answer in the following format:
{format_instructions}

<message>
{message}
</message>

Answer:
"""


class HatespeechDetectionFormat(BaseModel):
    explanation: str = Field(
        description="A comprehensive explanation of your classification in three sentences."
    )
    classification: str = Field(
        description="The classification of the message with one of the given categories. Only use the exact same words inside the XML category tags"
    )



HATESPEECH_DETECTOR_PROMPT = PromptTemplate(
    name="detector",
    template=_HATESPEECH_DETECTOR,
    input_variables=["message"],
    partial_variables={
        "format_instructions": JsonOutputParser(
            pydantic_object=HatespeechDetectionFormat
        ).get_format_instructions()
    },
)

### Hatespeech Validator -  Prompt build mainly with gpt4 (chatgpt plus)
_HATESPEECH_VALIDATOR = """You are tasked with re-evaluating the classification of a message that has been previously 
assessed by another expert in the context of hate speech detection. 
The message could fall into one of the following categories:

    `Direct hate speech`: The message contains hate speech expressed directly by the author. This does not include any personal experience where the author was the victim of hate speech.
    `Indirect hate speech`: The message contains hate speech that is not directly expressed by the author but implies endorsement or propagation of hate speech.
    `No hate speech`: The message does not exhibit any characteristics of hate speech.
    `Review needed`: The message exhibits some characteristics of hate speech but is not definitive, and therefore, further review is necessary.
    `Unknown`: The message could not be confidently classified due to ambiguity or lack of clear indicators.
    
Instructions: Carefully read the message, paying close attention to the context and the language used. Accurately apply the categories based on
the provided definitions, focusing on how hate speech is presented and the author's intent. Your detailed assessment is vital for ensuring a respectful
and safe communication environment. Always think step by step and decide based on your conclusion. Provide your perspective to the message and explain your conclusion.

Here is the classification provided by the previous expert below:
<opinion>    
Classification: {classification}
</opinion>    

Review the content of the message:
<message>
{message}
</message>

Always answer in the following format:
{format_instructions}

Answer:
"""


class HatespeechValidatorFormat(BaseModel):
    classification: str = Field(
        description="Your classification based on the opinion of the other expert"
    )
    explanation: str = Field(
        description="Your explanation if you agree or disagree with the expert's opinion in three sentence."
    )


HATESPEECH_VALIDATOR_PROMPT = PromptTemplate(
    name="validator",
    template=_HATESPEECH_VALIDATOR,
    input_variables=["classification", "explanation", "message"],
    partial_variables={
        "format_instructions": JsonOutputParser(
            pydantic_object=HatespeechValidatorFormat
        ).get_format_instructions()
    },
)

### Hatespeech subcategory classifier - Prompt build hybrid with gpt4 (chatgpt plus) and bigger refinements from my side.
_HATESPEECH_CLASSIFIER = """Please analyze the following message and categorize its content based on the listed categories.
Categorize the message in two major steps.

In the first step you categorize if one of the following main categories apply to the message:

    - Personal experience: Indicate if the message includes personal stories, anecdotes, or life experiences, where the author of the message suffered with hate speech.
    - Historical reference: Identify references or quotes to historical events, figures, or contexts. Only classify it as an `historical reference` if the historical content is used to provide information and is not used to insult specific groups or persons.
    - Offensive insult: The message clearly insults, discriminates or promoting hate and violence against particular groups, persons, and others.

In the second step you identify if the message contains any elements of racism, antisemitism, homophobia, ableism, sexism or other forms of hate speech.

Subcategories to analyze in the second step:

    - Racism: Identify any elements that promote, condone, or express prejudice, discrimination, or antagonism against a person or people based on their race or ethnic origin.
    - Antisemitism: Detect any expressions or actions that show hostility, prejudice, or discrimination against Jewish people.
    - Homophobia: Highlight statements or attitudes that express dislike of or prejudice against homosexual people.
    - Ableism: Point out any discrimination or social prejudice against people with disabilities or who are perceived to be disabled.
    - Violence: Indicate if the message promotes violence in any form or threatens particular groups, persons, and others.
    - Sexism: Highlight expressions or actions that show prejudice, discrimination, or antagonism based on gender, including demeaning statements, enforcement of traditional gender roles, or denial of opportunities due to gender.
    - Other hate speech: Identify any content that doesn't fall into the above categories but can be considered as promoting hate or discrimination against a particular group based on attributes such as gender, sexual orientation, religion, or others.

    
General instructions:
    - Analyze the content of the message carefully.
    - Categorize the second step with python booleans format and don't save it as a string.
    - If a subcategory does not apply, simply state false and move on to the next one.
    - Maintain an objective and unbiased stance throughout the analysis.
    - Always think step by step and decide based on your conclusion.

Always answer in the following format:
{format_instructions}

<message>
{message}
</message>

Answer:
"""


class HatespeechClassifierFormat(BaseModel):
    classification: str = Field(
        description="Categorize the message into the three main categories mentioned in the first step. Only use one category",
    )
    racism: bool = Field(
        description="If the content of the message applies to the subcategory `Racism` set it to True, else to False"
    )
    antisemitism: bool = Field(
        description="If the content of the message applies to the subcategory `Anitsemitism` set to True, else to False"
    )
    homophobia: bool = Field(
        description="If the content of the message applies to the subcategory `Homophobia` set to True, else to False"
    )
    ableism: bool = Field(
        description="If the content of the message applies to the subcategory `Ableism` set to True, else to False"
    )
    violence: bool = Field(
        description="If the content of the message applies to the subcategory `Violence` set to True, else to False"
    )
    sexism: bool = Field(
        description="If the content of the message applies to the subcategory `Sexism` set to True, else to False"
    )
    other_hate_speech: bool = Field(
        description="If the the message contains Other hate speech set to True, else to False"
    )
    explanation: str = Field(
        description="Provide a brief justification for each category you assign to the message, explaining how certain parts of the message meet the criteria for that category."
    )


HATESPEECH_CLASSIFICATION_PROMPT = PromptTemplate(
    name="classifier",
    template=_HATESPEECH_CLASSIFIER,
    input_variables=["message"],
    partial_variables={
        "format_instructions": JsonOutputParser(
            pydantic_object=HatespeechClassifierFormat
        ).get_format_instructions()
    },
)

# FCK NZS Rater - Prompt build hybrid with gpt4 and refinements from my side.
_RIGHT_WING_RATER = """You are an expert in understanding a broad spectrum of political ideologies, with a focus on
recognizing extreme and harmful rhetoric, including fascism, Nazism, and racial supremacy. You possess a deep
understanding of historical contexts, particularly related to Germany's history. Your task is to analyze messages in the first
step if the message contain any indication of right-wing ideologies and then rate them into four categories, based on the presence
and intensity of harmful right-wing ideologies.It is crucial to capture the intensity of the author's emotions, including feelings
of strong dislike or aversion. If the message describes situations where the author was the victim of right-wing ideologies, this
should be classified with no right-wing ideologies.

First step - Right-wing Indication:
    Classify with a boolean if the message contains any indication of right-wing ideologies. 

Second Step - Rating Scale:
    Category 0: The message contains no indication of right-wing ideologies.
    Category 1: The message contains subtle hints of right-wing ideologies but is not overtly harmful or inciting.
    Category 2: The message contains clear right-wing ideological rhetoric, promoting a divisive or exclusionary viewpoint, potentially harmful which needs be reported and further reviewed.
    Category 3: The message is extremely dangerous and constitutes a severe violation of German law. It openly advocates for violence, hatred, or supremacy based on right-wing extremist ideologies. Immediate action is required: reporting the content to the platform it was found on, and notifying law enforcement or other relevant authorities to assess the need for pressing charges.

Always think step by step and decide based on your conclusion. Also explain why and which part in the message was the most relevant for your conclusion.

Always answer in the following format:
{format_instructions}

<message>
{message}
</message>

Answer:
"""


class RightWingRatingFormat(BaseModel):
    right_wing_indicator: bool = Field(description="If the content of the message applies to the first step set it to True, else to False")
    rating: str = Field(description="The right wing rating into the four categories. Use only one of the mentioned categories")
    explanation: str = Field(description="A comprehensive explanation of your rating.")


RIGHT_WING_RATING_PROMPT = PromptTemplate(
    name="right-wing-rater",
    template=_RIGHT_WING_RATER,
    input_variables=["message"],
    partial_variables={
        "format_instructions": JsonOutputParser(
            pydantic_object=RightWingRatingFormat
        ).get_format_instructions()
    },
)


#
### Sarcasm Detector - maybe needed?
snippet = """Bad Humor: Determine if the message contains attempts at humor that are in poor taste, offensive,
or inappropriate, yet indicate that the user might not genuinely hold the opinion expressed in the joke."""
