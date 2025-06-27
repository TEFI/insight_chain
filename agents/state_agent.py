import json
from state.dialogue_state import DialogueState
from .base import BaseAgent
from jinja2 import Environment, FileSystemLoader


class StateAgent(BaseAgent):
    def __init__(self):
        super().__init__(temperature=0.8, top_p=0.95)

        # Start jinja enviroment
        self.env = Environment(loader=FileSystemLoader("prompts"))

        # load promt templates
        self.state_template = self.env.get_template("state_prompt_template.jinja2")

    def update_state(self, state: DialogueState) -> None:

        if not state.section_follow_up:
            state.section_index  += 1
            state.section = state.section_list[state.section_index]


            rendered_prompt = self.state_template.render(
                student_name=state.student.name,
                mentor_name=state.mentor.name,
                section=state.section_contents[state.section],
            )

            response = self.run(rendered_prompt)
            section_data = json.loads(response)

            section_follow_up = [
                f"**{list(d.values())[0]}**:\n{list(d.values())[0]}"
                for d in section_data
            ]
            state.section_follow_up = section_follow_up

