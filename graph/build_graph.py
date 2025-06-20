from langgraph.graph import StateGraph, END
from nodes.ask_node import ask_question_node
from nodes.answer_node import expert_node
from nodes.state_node import update_state_node
from nodes.evaluate_node import evaluator_node, evaluator_router
from state.dialogue_state import DialogueState

# Build and compile the dialogue state graph for the multi-agent system
def build_graph():
    # Initialize the graph with the DialogueState type
    graph_builder = StateGraph(DialogueState)

    # Define the entry point node where the dialogue begins
    graph_builder.set_entry_point("student_node")

    # Register each functional node in the dialogue cycle
    graph_builder.add_node("student_node", ask_question_node)
    graph_builder.add_node("mentor_node", expert_node)
    graph_builder.add_node("state_node", update_state_node)
    graph_builder.add_node("evaluator_node", evaluator_node)

    # Define the edges (transitions) between nodes
    graph_builder.add_edge("student_node", "evaluator_node")
    graph_builder.add_edge("mentor_node", "evaluator_node")
    graph_builder.add_conditional_edges("evaluator_node", evaluator_router)
    graph_builder.add_edge("state_node", END)  # End the flow after state update

    # Finalize and return the compiled graph
    return graph_builder.compile()

