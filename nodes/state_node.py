from agents.state_agent import StateAgent
from state.dialogue_state import DialogueState

# Initialize the agent responsible for managing state transitions
state_agent = StateAgent()

# Node function to update the current dialogue state
def update_state_node(state: DialogueState) -> DialogueState:
    # Delegate the update logic to the state agent
    state = state_agent.update_state(state)
    # Return the (mutated) state object
    return state
