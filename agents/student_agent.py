from .base import BaseAgent
from state.dialogue_state import DialogueState
from jinja2 import Environment, FileSystemLoader
from utils.template import TemplateWithSource

from evopromptfx.server.db.database import SessionLocal
from evopromptfx.server.tracker.prompt_tracker import PromptTracker


class StudentAgent(BaseAgent):
    def __init__(self):
        super().__init__(model_name="gemini-2.5-flash-preview-05-20", temperature=0.8, top_p=0.95)

        self.name = "studen_agent"

        # Setup Jinja2 environment and templates
        env = Environment(loader=FileSystemLoader("prompts"))
        self.templates = {
            "main": TemplateWithSource("student_prompt_template.jinja2", env, 1),
            "analogy": TemplateWithSource("student_analogy_prompt_template.jinja2", env, 1)
        }

        # Setup DB session and tracker
        self.db = SessionLocal()
        self.tracker = PromptTracker(self.db)

        self.experiment_id = self.tracker.create_experiment(
            name="Student Prompt Experiment",
            description="Tracking prompt versions and outputs from StudentAgent",
            created_by="student_agent"
        )

    def __del__(self):
        if hasattr(self, "db") and self.db:
            self.db.close()

    def __build_context(self, state: DialogueState) -> dict:
        return {
            "student_name": state.student.name,
            "mentor_name": state.mentor.name,
            "section": state.section,
            "context": state.context
        }

    def __render_prompt(self, template_key: str, **extra_fields) -> tuple[str, str]:
        base_context = extra_fields.pop("base_context")
        full_context = {**base_context, **extra_fields}
        rendered_prompt = self.templates[template_key].render(**full_context)
        return self.run(rendered_prompt), rendered_prompt

    def generate_question(self, state: DialogueState) -> str:
        context = self.__build_context(state)

        if state.section == state.section_list[0]:
            template_key = "analogy"
            token_estimate = str(int(len(state.context) * 0.1))
            context["tokens"] = token_estimate
        else:
            template_key = "main"
            context["memory"] = state.get_memory_as_text()
            context["mentor_answer"] = state.mentor.answer

        # Render and generate question
        response, rendered_prompt = self.__render_prompt(template_key, base_context=context)

        # Register prompt version
        version_id = self.tracker.start_prompt_version(
            experiment_id=self.experiment_id,
            name=f"student-{template_key}",
            template=self.templates[template_key].source,
            meta={"template_key": template_key},
            version=self.templates[template_key].version,
            environment="local",
            created_by=self.name
        )

        # Register output
        output_id = self.tracker.log_output(
            prompt_version_id=version_id,
            output_data=response,
            rendered_prompt=rendered_prompt,
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            created_by=self.name
        )

        # Update state
        state.student.rendered_prompt = rendered_prompt
        state.student.version_id = version_id
        state.student.output_id = output_id

        return response
