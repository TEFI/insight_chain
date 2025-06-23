from state.dialogue_state import DialogueState
from .base import BaseAgent
from jinja2 import Environment, FileSystemLoader
from evopromptfx.metrics.bert_score import compute_bertscore

import logging
from transformers import logging as hf_logging

class EvaluatorAgent(BaseAgent):
    def __init__(self, suppress_warnings=True):
        super().__init__(temperature=0.8, top_p=0.95)
        
        if suppress_warnings:
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            hf_logging.set_verbosity_error()
        

        # Start jinja enviroment
        # self.env = Environment(loader=FileSystemLoader("prompts"))

        # load promt templates
        # self.state_template = self.env.get_template("state_prompt_template.jinja2")


    def evaluate(self,  state: DialogueState) -> float:

        if state.last_node == "student_node":
            llm_response = state.student.question
        else:
            llm_response = state.mentor.answer

        score = compute_bertscore(state.context, llm_response)
        return score
