"""Prompt templates for Kappa operations."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PromptTemplate(BaseModel):
    """A reusable prompt template."""

    model_config = ConfigDict(frozen=True)

    name: str
    template: str
    description: str = ""
    variables: list[str] = Field(default_factory=list)

    def format(self, **kwargs: Any) -> str:
        """Format the template with provided values.

        Args:
            **kwargs: Template variable values.

        Returns:
            Formatted prompt string.
        """
        return self.template.format(**kwargs)


# System prompts
SYSTEM_PROMPT = PromptTemplate(
    name="system",
    description="Default system prompt for Claude sessions",
    template="""You are an expert software engineer working on the {project_name} project.

Your role is to implement high-quality, production-ready code following best practices.

Guidelines:
- Follow the project's code style and conventions
- Write clean, well-documented, maintainable code
- Include appropriate error handling and type hints
- Only modify files directly related to your assigned task
- Ask clarifying questions if requirements are unclear

Project Context:
{context}""",
    variables=["project_name", "context"],
)


TASK_PROMPT = PromptTemplate(
    name="task",
    description="Template for task execution prompts",
    template="""# Task: {task_name}

## Description
{task_description}

## Target Files
{file_targets}

## Dependencies
This task depends on: {dependencies}

## Instructions
1. Read any existing relevant files first
2. Implement the required functionality
3. Follow project code conventions
4. Add appropriate tests if applicable
5. Ensure code compiles and passes type checking

## Requirements
{requirements}

Begin implementation now.""",
    variables=[
        "task_name",
        "task_description",
        "file_targets",
        "dependencies",
        "requirements",
    ],
)


DECOMPOSITION_PROMPT = PromptTemplate(
    name="decomposition",
    description="Template for decomposing specifications into tasks",
    template="""Analyze the following project specification and decompose it into discrete, executable tasks.

## Specification
{specification}

## Output Format
For each task, provide:
1. **Name**: Short, descriptive name
2. **Description**: Detailed description of what needs to be done
3. **Category**: One of: setup, infrastructure, data_model, business_logic, api, ui, testing, documentation
4. **Complexity**: low, medium, or high
5. **Dependencies**: List of task names this depends on
6. **Files**: Target files to create or modify

## Guidelines
- Break down into small, focused tasks (1-2 hours of work each)
- Identify dependencies between tasks clearly
- Group related work together
- Include setup and testing tasks
- Order tasks to allow maximum parallelism

## Constraints
{constraints}

Provide the task breakdown in structured format.""",
    variables=["specification", "constraints"],
)


CONFLICT_RESOLUTION_PROMPT = PromptTemplate(
    name="conflict_resolution",
    description="Template for resolving code conflicts",
    template="""Two sessions have made conflicting changes to the same file.

## File
{file_path}

## Version A (Session {session_a})
```
{content_a}
```

## Version B (Session {session_b})
```
{content_b}
```

## Instructions
Analyze both versions and create a merged version that:
1. Preserves all intended functionality from both versions
2. Resolves any conflicts logically
3. Maintains code consistency and style
4. Does not introduce bugs or regressions

Explain your reasoning and provide the merged code.""",
    variables=["file_path", "session_a", "content_a", "session_b", "content_b"],
)


CODE_REVIEW_PROMPT = PromptTemplate(
    name="code_review",
    description="Template for reviewing generated code",
    template="""Review the following code for quality, correctness, and adherence to best practices.

## Code
```{language}
{code}
```

## Review Criteria
- Correctness: Does the code do what it's supposed to?
- Error handling: Are errors handled appropriately?
- Type safety: Are type hints present and correct?
- Code style: Does it follow conventions?
- Performance: Are there any obvious inefficiencies?
- Security: Are there any security concerns?

Provide specific feedback and suggestions for improvement.""",
    variables=["language", "code"],
)


CONTEXT_SUMMARY_PROMPT = PromptTemplate(
    name="context_summary",
    description="Template for summarizing project context",
    template="""Summarize the following project context for use by other development sessions.

## Decisions Made
{decisions}

## Files Analyzed
{files}

## Key Discoveries
{discoveries}

Create a concise summary that captures:
1. Important architectural decisions and their rationale
2. Key patterns and conventions observed
3. Dependencies and relationships between components
4. Any gotchas or important notes for developers

Keep the summary under 500 words while preserving critical information.""",
    variables=["decisions", "files", "discoveries"],
)
