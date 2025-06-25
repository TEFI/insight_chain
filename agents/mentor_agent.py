from jinja2 import Environment, FileSystemLoader
from state.dialogue_state import DialogueState
from .base import BaseAgent

class MentorAgent(BaseAgent):
    def __init__(self):
        super().__init__(temperature=0.8, top_p=0.95)
        env = Environment(loader=FileSystemLoader("prompts"))
        self.templates = {
            "main": env.get_template("mentor_prompt_template.jinja2"),
            "analogy": env.get_template("mentor_analogy_prompt_template.jinja2")
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
        prompt = self.templates[template_key].render(full_context)
        return self.run(prompt), prompt

    def answer_question(self, state: DialogueState) -> str:
        context = self.__build_context(state)

        if state.section == state.section_list[0]:
            response, prompt = self.__render_prompt("analogy", base_context=context)
        else:
            memory_text = state.get_memory_as_text()
            response, prompt = self.__render_prompt("main", base_context=context, memory=memory_text)

        state.mentor.prompt = prompt
        return response
