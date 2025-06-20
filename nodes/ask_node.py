from agents.student_agent import StudentAgent
from state.dialogue_state import DialogueState
from constants.node_labels import STUDENT_NODE

agent = StudentAgent()

def ask_question_node(state: DialogueState) -> DialogueState:
    # Generate the student's question using the agent
    generated_question = agent.generate_question(state)
    # Update the student's question field directly
    state.student.question = generated_question
    # Set the last_node field
    state.last_node = STUDENT_NODE
    # Return the (mutated) state object
    return state
