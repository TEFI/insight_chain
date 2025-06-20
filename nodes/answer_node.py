from agents.mentor_agent import MentorAgent
from state.dialogue_state import DialogueState
from constants.node_labels import MENTOR_NODE

mentor = MentorAgent()

def expert_node(state: DialogueState) -> DialogueState:
    # Generate the mentor's answer using the agent
    generated_answer = mentor.answer_question(state)
    # Update the mentor's answer field directly
    state.mentor.answer = generated_answer
    # Set the last_node field
    state.last_node = MENTOR_NODE
    # Return the (mutated) state object
    return state
