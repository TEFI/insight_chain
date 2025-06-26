from .base import BaseAgent
from state.dialogue_state import DialogueState
from jinja2 import Environment, FileSystemLoader
from utils.template import TemplateWithSource

class StudentAgent(BaseAgent):
    def __init__(self):
        super().__init__(temperature=0.8, top_p=0.95)
        env = Environment(loader=FileSystemLoader("prompts"))
        self.templates = {
            "main": TemplateWithSource("student_prompt_template.jinja2", env),
            "analogy": TemplateWithSource("student_analogy_prompt_template.jinja2", env)
        }

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
            token_estimate = str(int(len(state.context) * 0.1))
            state.student.prompt = self.templates["analogy"].source
            response, rendered_prompt = self.__render_prompt("analogy", base_context=context, tokens=token_estimate)
        else:
            memory_text = state.get_memory_as_text()
            state.student.prompt = self.templates["main"].source
            response, rendered_prompt = self.__render_prompt(
                "main",
                base_context=context,
                memory=memory_text,
                mentor_answer=state.mentor.answer
            )

        state.student.rendered_prompt = rendered_prompt
        return response
