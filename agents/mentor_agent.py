from jinja2 import Environment, FileSystemLoader

from state.dialogue_state import DialogueState
from .base import BaseAgent

class MentorAgent(BaseAgent):
    def __init__(self):
        super().__init__(temperature=0.8, top_p=0.95)

        # Start jinja enviroment
        self.env = Environment(loader=FileSystemLoader("prompts"))

        # load promt templates
        self.prompt_template = self.env.get_template("mentor_prompt_template.jinja2")
        self.analogy_template = self.env.get_template("mentor_analogy_prompt_template.jinja2")

    def __answer_analogy(self, context, section, student_name, mentor_name, document_title, student_question):
        rendered_prompt = self.analogy_template.render(
            student_name=student_name,
            mentor_name=mentor_name,
            document_title=document_title,
            context=context,
            section=section,
            student_question=student_question
        )
        return self.run(rendered_prompt)

    def answer_question(self,  state: DialogueState) -> str:
        section = state.section
        context = state.context
        mentor_name = state.mentor.name
        student_name = state.student.name
        document_title = state.document_title
        student_question = state.student.question

        if section == state.section_list[0]: 
            return self.__answer_analogy(context, section, student_name, mentor_name, document_title, student_question)

        memory_text = state.get_memory_as_text()

        rendered_prompt = self.prompt_template.render(
            student_name=student_name,
            section=section,
            mentor_name=mentor_name,
            context=context,
            student_question=student_question,
            memory=memory_text
        )

        return self.run(rendered_prompt)
