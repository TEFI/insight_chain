from state.dialogue_state import DialogueState
from .base import BaseAgent
from config import GEMINI_API_KEY
from evopromptfx.core.judges.gemini_judge import GeminiJudge

class EvaluatorAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.llm_as_judge = GeminiJudge(api_key=GEMINI_API_KEY)
    def evaluate(self,  state: DialogueState) -> float:

        if state.last_node == "student_node":
            llm_response = state.student.question
            prompt = state.student.prompt
        else:
            llm_response = state.mentor.answer
            prompt = state.mentor.prompt

        score = self.llm_as_judge.evaluate(
            prompt=prompt,
            answer=llm_response
        )

        return score["average"]

