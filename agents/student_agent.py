from .base import BaseAgent
from state.dialogue_state import DialogueState
from jinja2 import Environment, FileSystemLoader

class StudentAgent(BaseAgent):
    def __init__(self):
        super().__init__(temperature=0.8, top_p=0.95)

        # Initialize Jinja2 environment
        self.env = Environment(loader=FileSystemLoader("prompts"))

        # Load prompt templates
        self.prompt_template = self.env.get_template("student_prompt_template.jinja2")
        self.analogy_template = self.env.get_template("student_analogy_prompt_template.jinja2")

    def __generate_analogy(self, context: str, section: str, student_name: str, mentor_name: str) -> str:
        rendered_prompt = self.analogy_template.render(
            student_name=student_name,
            context=context,
            tokens=str(int(len(context) * 0.1)),
            section=section,
            mentor_name=mentor_name
        )
        return self.run(rendered_prompt)

    def generate_question(self, state: DialogueState) -> str:
        section = state.section
        context = state.context
        mentor_name = state.mentor.name
        student_name = state.student.name

        if section == state.section_list[0]:  # first section
            return self.__generate_analogy(context, section, student_name, mentor_name)

        memory_text = state.get_memory_as_text()
        mentor_answer = state.mentor.answer

        rendered_prompt = self.prompt_template.render(
            student_name=student_name,
            section=section,
            mentor_name=mentor_name,
            context=context,
            mentor_answer=mentor_answer,
            memory=memory_text
        )

        return self.run(rendered_prompt)
