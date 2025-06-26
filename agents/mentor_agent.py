from jinja2 import Environment, FileSystemLoader
from state.dialogue_state import DialogueState
from .base import BaseAgent
from utils.template import TemplateWithSource


class MentorAgent(BaseAgent):
    def __init__(self):
        super().__init__(temperature=0.8, top_p=0.95)
        env = Environment(loader=FileSystemLoader("prompts"))

        self.templates = {
            "main": TemplateWithSource("mentor_prompt_template.jinja2", env),
            "analogy": TemplateWithSource("mentor_analogy_prompt_template.jinja2", env)
        }

    def __build_context(self, state: DialogueState) -> dict:
        return {
            "student_name": state.student.name,
            "mentor_name": state.mentor.name,
            "document_title": state.document_title,
            "context": state.context,
            "section": state.section,
            "student_question": state.student.question,
        }

    def __render_prompt(self, template_key: str, **extra_fields) -> tuple[str, str]:
        base_context = extra_fields.pop("base_context")
        full_context = {**base_context, **extra_fields}
        rendered_prompt = self.templates[template_key].render(**full_context)
        return self.run(rendered_prompt), rendered_prompt

    def answer_question(self, state: DialogueState) -> str:
        context = self.__build_context(state)

        if state.section == state.section_list[0]:
            state.mentor.prompt = self.templates["analogy"].source
            response, rendered_prompt = self.__render_prompt("analogy", base_context=context)
        else:
            memory_text = state.get_memory_as_text()
            state.mentor.prompt = self.templates["main"].source
            response, rendered_prompt = self.__render_prompt("main", base_context=context, memory=memory_text)

        state.mentor.rendered_prompt = rendered_prompt
        return response
