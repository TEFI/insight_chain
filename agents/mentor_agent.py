from jinja2 import Environment, FileSystemLoader
from state.dialogue_state import DialogueState
from .base import BaseAgent
from utils.template import TemplateWithSource

from evopromptfx.server.db.database import SessionLocal
from evopromptfx.server.tracker.prompt_tracker import PromptTracker


class MentorAgent(BaseAgent):
    def __init__(self):
        super().__init__(model_name="gemini-2.5-flash-preview-05-20", temperature=0.8, top_p=0.95)

        self.name = "mentor_agent"

        # Load prompt templates
        env = Environment(loader=FileSystemLoader("prompts"))
        self.templates = {
            "main": TemplateWithSource("mentor_prompt_template.jinja2", env, 1),
            "analogy": TemplateWithSource("mentor_analogy_prompt_template.jinja2", env, 1),
        }

        # Setup DB connection and prompt tracker
        self.db = SessionLocal()
        self.tracker = PromptTracker(self.db)

        # Register experiment
        self.experiment_id = self.tracker.create_experiment(
            name="Mentor Prompt Experiment",
            description="Tracking prompt versions and outputs from MentorAgent",
            created_by=self.name
        )

    def __del__(self):
        if hasattr(self, "db") and self.db:
            self.db.close()

    def __build_context(self, state: DialogueState) -> dict:
        context = {
            "student_name": state.student.name,
            "mentor_name": state.mentor.name,
            "document_title": state.document_title,
            "context": state.context,
            "section": state.section,
            "student_question": state.student.question,
        }
        return context

    def __render_prompt(self, template_key: str, **extra_fields) -> tuple[str, str]:
        base_context = extra_fields.pop("base_context")
        full_context = {**base_context, **extra_fields}
        rendered_prompt = self.templates[template_key].render(**full_context)
        return self.run(rendered_prompt), rendered_prompt

    def answer_question(self, state: DialogueState) -> str:
        context = self.__build_context(state)

        template_key = "analogy" if state.section == state.section_list[0] else "main"
        if template_key == "main":
            context["memory"] = state.get_memory_as_text()
            context["key_points"] = "\n\n".join(state.section_follow_up)
            context["actual_point"] = state.section_follow_up[0]

        response, rendered_prompt = self.__render_prompt(template_key, base_context=context)

        version_id = self.tracker.start_prompt_version(
            experiment_id=self.experiment_id,
            name=f"{self.name}-{template_key}",
            template=self.templates[template_key].source,
            meta={"template_key": template_key},
            version=self.templates[template_key].version,
            environment="local",
            created_by=self.name
        )

        output_id = self.tracker.log_output(
            prompt_version_id=version_id,
            output_data=response,
            rendered_prompt=rendered_prompt,
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            created_by=self.name
        )

        state.mentor.version_id = version_id
        state.mentor.output_id = output_id
        state.mentor.rendered_prompt = rendered_prompt

        return response
