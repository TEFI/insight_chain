from agents.evaluator_agent import EvaluatorAgent
from state.dialogue_state import DialogueState
from constants.node_labels import STUDENT_NODE, MENTOR_NODE, STATE_NODE
from constants.evaluation import MIN_ACCEPTABLE_SCORE

# Initialize the evaluator agent responsible for scoring the interaction
evaluator_agent = EvaluatorAgent()

# Node that evaluates the quality of the last exchange
def evaluator_node(state: DialogueState) -> DialogueState:
    # Compute the evaluation score for the current state
    score = evaluator_agent.evaluate(state)
    state.score = score
    origin = state.last_node

    # If the mentor's answer was good enough, record the turn in memory
    if score >= MIN_ACCEPTABLE_SCORE and origin == MENTOR_NODE:
        state.add_turn(state.student.question, state.mentor.answer)

    return state

# Router that decides the next node to run based on score and origin
def evaluator_router(state: DialogueState) -> str:
    score = state.score
    origin = state.last_node

    valid_nodes = {STUDENT_NODE, MENTOR_NODE}

    if origin not in valid_nodes:
        raise ValueError(f"Unknown origin: {origin}")

    # If the score is too low, retry the same node
    if score < MIN_ACCEPTABLE_SCORE:
        return origin

    # If the score is acceptable, move to the next step
    return {
        STUDENT_NODE: MENTOR_NODE,
        MENTOR_NODE: STATE_NODE
    }[origin]
