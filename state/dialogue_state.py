from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional

# Represents a single dialogue turn between student and mentor
class DialogueTurn(BaseModel):
    student_question: str
    mentor_answer: str

    # Format the dialogue turn as a string
    def as_text(self, student_name: str, mentor_name: str) -> str:
        return f"{student_name}: {self.student_question}\n{mentor_name}: {self.mentor_answer}\n"


# Stores a memory of recent dialogue turns, limited by max_turns
class Memory(BaseModel):
    turns: List[DialogueTurn] = Field(default_factory=list)
    max_turns: int = 4

    # Add a new turn to memory and trim if exceeding max_turns
    def add(self, question: str, answer: str) -> None:
        new = DialogueTurn(
            student_question=question.strip(),
            mentor_answer=answer.strip()
        )
        self.turns.append(new)
        self.turns = self.turns[-self.max_turns:]

    # Convert the memory into a formatted string dialogue
    def as_text(self, student_name: str, mentor_name: str) -> str:
        return "".join(turn.as_text(student_name, mentor_name) for turn in self.turns)


# Stores agent data for student or mentor
class AgentData(BaseModel):
    name: str
    question: Optional[str] = None
    answer: Optional[str] = None
    version_id: Optional[str] = None
    rendered_prompt: Optional[str] = None
    output_id: Optional[str] = None


# Represents the global dialogue state across graph execution
class DialogueState(BaseModel):
    model_config = ConfigDict(frozen=False)

    document_title: str
    section_contents: dict
    section_list: list[str] = Field(default_factory=list)
    section_index: int = 0
    student: AgentData
    mentor: AgentData
    score: float = 0
    last_node: Optional[str] = None
    memory: Memory = Field(default_factory=Memory)
    context: Optional[str] = None
    section: Optional[str] = None
    section_follow_up: list[str] = Field(default_factory=list)
    # Add a completed dialogue turn to memory
    def add_turn(self, question: str, answer: str) -> None:
        self.memory.add(question, answer)

    # Get all memory turns formatted as string
    def get_memory_as_text(self) -> str:
        return self.memory.as_text(self.student.name, self.mentor.name)

    # Factory method to create the initial dialogue state
    @classmethod
    def create_initial_state(
        cls,
        student_name: str,
        mentor_name: str,
        section_contents: dict,
        document_title: str
    ) -> DialogueState:
        section_list = list(section_contents.keys())
        return cls(
            student=AgentData(name=student_name),
            mentor=AgentData(name=mentor_name),
            document_title=document_title,
            section_contents=section_contents,
            section_list=section_list,
            section_index=0,
            section=section_list[0],
            context=section_contents[section_list[0]]
        )
