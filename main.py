import argparse
from pathlib import Path

from graph.build_graph import build_graph
from state.dialogue_state import DialogueState
from utils.parser import PaperParser

# Ensure the returned state from the graph is always a DialogueState instance.
# Accepts either an actual DialogueState object or a dictionary of attributes.
def ensure_dialogue_state(state_like) -> DialogueState:
    if isinstance(state_like, DialogueState):
        return state_like
    elif isinstance(state_like, dict):
        return DialogueState(**state_like)
    raise TypeError("Invalid state returned from graph")

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run multiagent document parser")
    parser.add_argument(
        "--doc",
        type=Path,
        #required=True
        default=Path("papers/zero/zero.md")
        
    )
    args = parser.parse_args()

    # Load and parse the paper into structured sections
    document_path = args.doc
    parser = PaperParser(document_path)
    section_contents, document_title = parser.parse_sections()

    # Initialize the dialogue state with context and participants
    state = DialogueState.create_initial_state(
        document_title=document_title,
        section_contents=section_contents,
        student_name="Maria",
        mentor_name="Roberto"
    )

    # Build the multi-agent reasoning graph
    graph = build_graph()

    # Run the graph iteratively through the document sections
    while (state.section_index < len(state.section_list)) and state.section_follow_up:
        try:
            raw_output = graph.invoke(state)
            state = ensure_dialogue_state(raw_output)

            print(f"{state.student.name}: {state.student.question}\n\n")
            print(f"{state.mentor.name}: {state.mentor.answer}\n\n")

        except Exception as e:
            print(f"❌ Error during graph invocation: {e}")
            break
