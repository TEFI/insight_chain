from state.dialogue_state import DialogueState
from .base import BaseAgent
from config import GEMINI_API_KEY
from evopromptfx.core.judges.gemini_judge import GeminiJudge

from evopromptfx.server.db.database import SessionLocal
from evopromptfx.server.tracker.prompt_tracker import PromptTracker


class EvaluatorAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.name = "evaluator_agent"
        self.llm_as_judge = GeminiJudge(api_key=GEMINI_API_KEY)

        # DB session and tracker
        self.db = SessionLocal()
        self.tracker = PromptTracker(self.db)

    def __del__(self):
        if hasattr(self, "db") and self.db:
            self.db.close()

    def evaluate(self, state: DialogueState) -> float:
        # Get agent's state
        agent_state = state.student if state.last_node == "student_node" else state.mentor

        # Extract necessary fields
        output_data = agent_state.question if state.last_node == "student_node" else agent_state.answer
        rendered_prompt = agent_state.rendered_prompt
        prompt_version_id = agent_state.version_id
        output_id = agent_state.output_id

        # Evaluate using LLM-as-a-Judge
        metrics = self.llm_as_judge.evaluate(
            prompt=rendered_prompt,
            answer=output_data
        )

        # Register evaluation
        self.tracker.log_evaluation(
            prompt_version_id=prompt_version_id,
            output_id=output_id,
            model_name=self.model_name,
            metrics=metrics,
            created_by=self.name
        )

        return metrics["average"]
