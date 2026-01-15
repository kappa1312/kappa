"""
Prompt templates for Kappa operations.

This module provides pre-built prompt templates for various Kappa operations
including task execution, parallel execution, validation, and context sharing.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# TEMPLATE MODEL
# =============================================================================


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

    def get_missing_variables(self, **kwargs: Any) -> list[str]:
        """Get list of variables not provided.

        Args:
            **kwargs: Provided variables.

        Returns:
            List of missing variable names.
        """
        return [v for v in self.variables if v not in kwargs]


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================


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


PARALLEL_SYSTEM_PROMPT = PromptTemplate(
    name="parallel_system",
    description="System prompt for parallel task execution sessions",
    template="""You are an expert software engineer working on the {project_name} project as part of an autonomous development team.

Your current task is being executed in parallel with other tasks. This requires:
- Strict adherence to your assigned files only
- No modifications to shared configuration files without explicit instruction
- Clear, self-contained implementations

Guidelines:
- Follow the project's code style and conventions
- Write clean, well-documented, maintainable code
- Include appropriate error handling and type hints
- ONLY modify files explicitly assigned to your task
- Do not touch files being modified by parallel tasks

Tech Stack: {tech_stack}

Project Context:
{context}""",
    variables=["project_name", "tech_stack", "context"],
)


# =============================================================================
# TASK PROMPTS
# =============================================================================


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


PARALLEL_TASK_PROMPT = PromptTemplate(
    name="parallel_task",
    description="Template for parallel task execution with context awareness",
    template="""# Task: {task_title}
**Wave**: {wave_number} | **Category**: {category} | **Priority**: {priority}

## Description
{task_description}

## Files to Create
{files_to_create}

## Files to Modify
{files_to_modify}

## Available Types
{type_definitions}

## Available Imports
{imports}

## Dependencies
{dependencies}

## Context from Previous Tasks
{context_from_tasks}

## Parallel Execution Notice
This task is executing in **Wave {wave_number}** alongside:
{parallel_tasks}

**IMPORTANT**: Only modify your assigned files. Do not touch files being modified by parallel tasks.

## Instructions
1. Read any existing relevant files first
2. Implement the required functionality completely
3. Follow project code conventions strictly
4. Include proper error handling
5. Add type hints to all functions
6. Export any types/interfaces that other tasks may need

## Validation
After completing, run:
{validation_commands}

Execute this task completely. Create production-quality code.""",
    variables=[
        "task_title",
        "wave_number",
        "category",
        "priority",
        "task_description",
        "files_to_create",
        "files_to_modify",
        "type_definitions",
        "imports",
        "dependencies",
        "context_from_tasks",
        "parallel_tasks",
        "validation_commands",
    ],
)


# =============================================================================
# DECOMPOSITION PROMPTS
# =============================================================================


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


ENHANCED_DECOMPOSITION_PROMPT = PromptTemplate(
    name="enhanced_decomposition",
    description="Enhanced decomposition with project type awareness",
    template="""Analyze the following project specification and decompose it into atomic, parallelizable tasks.

## Project Details
- **Name**: {project_name}
- **Type**: {project_type}
- **Tech Stack**: {tech_stack}

## Specification
{specification}

## Features to Implement
{features}

## Output Format (JSON)
For each task, provide:
```json
{{
    "id": "unique-task-id",
    "title": "Short descriptive title",
    "description": "Detailed implementation instructions",
    "category": "setup|infrastructure|data_model|business_logic|api|ui|testing|documentation",
    "complexity": "low|medium|high",
    "session_type": "terminal|web|native",
    "dependencies": ["task-id-1", "task-id-2"],
    "files_to_create": ["path/to/new/file.ts"],
    "files_to_modify": ["path/to/existing/file.ts"],
    "requires_context_from": ["task-id-that-exports-types"],
    "validation_commands": ["npm run typecheck", "npm test"],
    "estimated_duration_minutes": 30
}}
```

## Wave Organization Guidelines
- **Wave 0**: Project setup, configuration, package.json
- **Wave 1**: Type definitions, interfaces, base models
- **Wave 2**: Core services, utilities, helpers
- **Wave 3**: Business logic, API routes, components
- **Wave 4**: Integration, pages, main features
- **Wave 5**: Testing, documentation, polish

## Parallelism Guidelines
- Tasks in the same wave execute in parallel
- Minimize file conflicts within waves
- Each task should be self-contained
- Export types/interfaces for dependent tasks

## Constraints
{constraints}

Provide the complete task breakdown as a JSON array.""",
    variables=[
        "project_name",
        "project_type",
        "tech_stack",
        "specification",
        "features",
        "constraints",
    ],
)


# =============================================================================
# VALIDATION PROMPTS
# =============================================================================


VALIDATION_PROMPT = PromptTemplate(
    name="validation",
    description="Template for validating project outputs",
    template="""Validate the project outputs in the workspace.

## Workspace
{workspace}

## Tasks Completed
{task_count} tasks have been executed.

## Files Created
{files_created}

## Files Modified
{files_modified}

## Validation Checklist
1. **Type Safety**: Run type checker and verify no errors
2. **Build**: Verify the project builds successfully
3. **Tests**: Run test suite and check for failures
4. **Lint**: Check for code style violations
5. **Integration**: Verify components work together

## Instructions
1. Run the validation commands for this project
2. Report any failures or issues found
3. Suggest fixes for any problems
4. Confirm the project is production-ready

Provide a detailed validation report.""",
    variables=["workspace", "task_count", "files_created", "files_modified"],
)


VALIDATION_REPORT_PROMPT = PromptTemplate(
    name="validation_report",
    description="Template for generating validation reports",
    template="""Generate a validation report for the completed project.

## Project: {project_name}
## Workspace: {workspace}

## Execution Summary
- Total Tasks: {total_tasks}
- Completed: {completed_tasks}
- Failed: {failed_tasks}
- Waves Executed: {waves_executed}

## Files Summary
### Created ({files_created_count})
{files_created}

### Modified ({files_modified_count})
{files_modified}

## Validation Results
### Type Check
{typecheck_result}

### Build
{build_result}

### Tests
{test_result}

### Lint
{lint_result}

## Issues Found
{issues}

## Recommendations
Based on the validation results, provide:
1. Critical issues that must be fixed
2. Warnings that should be addressed
3. Suggestions for improvement
4. Overall project health assessment

Generate a comprehensive report.""",
    variables=[
        "project_name",
        "workspace",
        "total_tasks",
        "completed_tasks",
        "failed_tasks",
        "waves_executed",
        "files_created_count",
        "files_created",
        "files_modified_count",
        "files_modified",
        "typecheck_result",
        "build_result",
        "test_result",
        "lint_result",
        "issues",
    ],
)


# =============================================================================
# CONTEXT PROMPTS
# =============================================================================


CONTEXT_INJECTION_PROMPT = PromptTemplate(
    name="context_injection",
    description="Template for injecting context mid-task",
    template="""## Context Update

New context has been discovered from parallel tasks that may be relevant:

### New Type Definitions
{type_definitions}

### New Exports Available
{exports}

### Files Created by Parallel Tasks
{files_created}

### Important Decisions
{decisions}

Incorporate this context into your current work as needed.
Continue with your task implementation.""",
    variables=["type_definitions", "exports", "files_created", "decisions"],
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


WAVE_CONTEXT_PROMPT = PromptTemplate(
    name="wave_context",
    description="Template for providing context from previous waves",
    template="""## Context from Previous Waves

### Wave {previous_wave} Outputs

The following was completed in the previous wave:

#### Tasks Completed
{completed_tasks}

#### Files Created
{files_created}

#### Types/Interfaces Exported
{exported_types}

#### Key Decisions Made
{decisions}

### Available for Import
{available_imports}

Use this context to inform your implementation. Import types and utilities
as needed from the files created in previous waves.""",
    variables=[
        "previous_wave",
        "completed_tasks",
        "files_created",
        "exported_types",
        "decisions",
        "available_imports",
    ],
)


# =============================================================================
# CONFLICT RESOLUTION PROMPTS
# =============================================================================


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


CONFLICT_ANALYSIS_PROMPT = PromptTemplate(
    name="conflict_analysis",
    description="Template for analyzing potential conflicts before execution",
    template="""Analyze the following tasks for potential conflicts.

## Wave {wave_number} Tasks

{tasks}

## Conflict Analysis Required
1. **File Conflicts**: Do any tasks modify the same files?
2. **Import Conflicts**: Do tasks export conflicting type definitions?
3. **State Conflicts**: Do tasks modify shared state or configuration?
4. **Order Dependencies**: Are there implicit ordering requirements?

## Output Format
```json
{{
    "conflicts": [
        {{
            "type": "file|import|state|order",
            "tasks": ["task-id-1", "task-id-2"],
            "description": "Description of the conflict",
            "severity": "critical|warning|info",
            "resolution": "Suggested resolution"
        }}
    ],
    "safe_to_parallelize": true|false,
    "recommendations": ["recommendation 1", "recommendation 2"]
}}
```

Provide a thorough conflict analysis.""",
    variables=["wave_number", "tasks"],
)


# =============================================================================
# CODE REVIEW PROMPTS
# =============================================================================


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


TASK_OUTPUT_REVIEW_PROMPT = PromptTemplate(
    name="task_output_review",
    description="Template for reviewing task execution output",
    template="""Review the output from task execution.

## Task: {task_title}
## Session ID: {session_id}
## Duration: {duration_seconds}s

## Files Created
{files_created}

## Files Modified
{files_modified}

## Session Output
```
{output}
```

## Errors/Warnings
{errors}

## Review Questions
1. Did the task complete successfully?
2. Were all required files created/modified?
3. Are there any errors that need attention?
4. Is the implementation complete and correct?

Provide a task completion assessment.""",
    variables=[
        "task_title",
        "session_id",
        "duration_seconds",
        "files_created",
        "files_modified",
        "output",
        "errors",
    ],
)


# =============================================================================
# ERROR HANDLING PROMPTS
# =============================================================================


ERROR_RECOVERY_PROMPT = PromptTemplate(
    name="error_recovery",
    description="Template for recovering from task execution errors",
    template="""A task has failed and needs recovery.

## Failed Task: {task_title}
## Error Type: {error_type}
## Error Message:
```
{error_message}
```

## Stack Trace
```
{stack_trace}
```

## Task Context
- Files being created: {files_to_create}
- Files being modified: {files_to_modify}
- Dependencies: {dependencies}

## Partial Output
```
{partial_output}
```

## Recovery Instructions
1. Analyze the error and identify the root cause
2. Determine if the task can be retried
3. If retry is possible, fix the issue and continue
4. If not, document what was completed and what remains

Provide error analysis and recovery steps.""",
    variables=[
        "task_title",
        "error_type",
        "error_message",
        "stack_trace",
        "files_to_create",
        "files_to_modify",
        "dependencies",
        "partial_output",
    ],
)


RETRY_PROMPT = PromptTemplate(
    name="retry",
    description="Template for retrying a failed task",
    template="""Retrying task after previous failure.

## Task: {task_title}
## Attempt: {attempt_number} of {max_attempts}

## Previous Error
{previous_error}

## What Was Completed
{completed_steps}

## What Remains
{remaining_steps}

## Modified Approach
Based on the previous error, the following adjustments should be made:
{adjustments}

## Instructions
1. Do not repeat work that was already completed
2. Apply the modified approach to avoid the previous error
3. Complete the remaining steps
4. Validate the final output

Continue with the task using the modified approach.""",
    variables=[
        "task_title",
        "attempt_number",
        "max_attempts",
        "previous_error",
        "completed_steps",
        "remaining_steps",
        "adjustments",
    ],
)


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


# All templates for easy access
ALL_TEMPLATES: dict[str, PromptTemplate] = {
    # System
    "system": SYSTEM_PROMPT,
    "parallel_system": PARALLEL_SYSTEM_PROMPT,
    # Task
    "task": TASK_PROMPT,
    "parallel_task": PARALLEL_TASK_PROMPT,
    # Decomposition
    "decomposition": DECOMPOSITION_PROMPT,
    "enhanced_decomposition": ENHANCED_DECOMPOSITION_PROMPT,
    # Validation
    "validation": VALIDATION_PROMPT,
    "validation_report": VALIDATION_REPORT_PROMPT,
    # Context
    "context_injection": CONTEXT_INJECTION_PROMPT,
    "context_summary": CONTEXT_SUMMARY_PROMPT,
    "wave_context": WAVE_CONTEXT_PROMPT,
    # Conflict
    "conflict_resolution": CONFLICT_RESOLUTION_PROMPT,
    "conflict_analysis": CONFLICT_ANALYSIS_PROMPT,
    # Review
    "code_review": CODE_REVIEW_PROMPT,
    "task_output_review": TASK_OUTPUT_REVIEW_PROMPT,
    # Error
    "error_recovery": ERROR_RECOVERY_PROMPT,
    "retry": RETRY_PROMPT,
}


def get_template(name: str) -> PromptTemplate | None:
    """Get a template by name.

    Args:
        name: Template name.

    Returns:
        PromptTemplate if found, None otherwise.
    """
    return ALL_TEMPLATES.get(name)


def list_templates() -> list[str]:
    """List all available template names.

    Returns:
        List of template names.
    """
    return list(ALL_TEMPLATES.keys())
