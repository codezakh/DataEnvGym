from dataenvgym.gym.domain_models import CodeGenerationTaskInstance
from . import tool_api
from .constants import CODE_DEMO_EXAMPLES
import jinja2


def load_tool_descriptions() -> str:
    # Get the source code of the tool_api module
    with open(tool_api.__file__, "r") as f:
        source_code = f.read()
    # There is a premable that needs to be stripped.
    # It starts with # BEGIN PREAMBLE # and ends with # END PREAMBLE #.
    # Everything in between is a comment and should be stripped.
    lines = source_code.split("\n")
    kept_lines: list[str] = []
    is_in_preamble = True
    line_iterator = iter(lines)
    while is_in_preamble:
        line = next(line_iterator)
        # If we reach the end of the preamble, add the rest of the lines to the kept lines
        if line.startswith("# END PREAMBLE #"):
            is_in_preamble = False
            # Add the rest of the lines to the kept lines
            kept_lines.extend(line_iterator)
        else:
            # This is a comment line of the preamble so we can ignore it
            continue
    return "\n".join(kept_lines)


SINGLE_TURN_ANSWER_PROMPT_TEMPLATE = jinja2.Template(
    """# Tool Description
{{ tool_descriptions }}

{% if examples %}
# Examples
{% for example in examples %}
User Request: {{ example.user_request }}
Response:
```python
{{ example.prediction }}
```
{% endfor %}
{% endif %}

# Instruction
Use the given tools (in Tool Description) to solve the user request by writing Python code.
Follow the format shown in the examples.
Surround your response with ```python and ``` to be a valid Python code block.

User Request: {{ user_request }}
Response:
""",
    undefined=jinja2.StrictUndefined,
)


def make_single_turn_inference_prompt_from_instruction(
    instruction: str,
) -> str:
    tool_descriptions = load_tool_descriptions()
    return SINGLE_TURN_ANSWER_PROMPT_TEMPLATE.render(
        tool_descriptions=tool_descriptions,
        examples=CODE_DEMO_EXAMPLES,
        user_request=instruction,
    )


def make_single_turn_inference_prompt_from_task_instance(
    task_instance: CodeGenerationTaskInstance,
) -> str:
    return make_single_turn_inference_prompt_from_instruction(task_instance.instruction)
