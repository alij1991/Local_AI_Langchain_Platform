"""Pre-built agent system templates for one-click deployment."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SystemTemplate:
    id: str
    name: str
    description: str
    icon: str  # Material icon name for Flutter
    category: str  # research | coding | writing | general | creative | data
    system_prompt: str
    tool_ids: list[str] = field(default_factory=list)
    recommended_models: list[str] = field(default_factory=list)
    default_settings: dict = field(default_factory=dict)


SYSTEM_TEMPLATES: list[SystemTemplate] = [
    SystemTemplate(
        id="research_assistant",
        name="Research Assistant",
        description="Thorough researcher that finds, verifies, and synthesizes information from the web.",
        icon="science",
        category="research",
        system_prompt="""You are a thorough research assistant. Your job is to help the user find, verify, and synthesize information.

Guidelines:
- Always search the web for current information before answering factual questions
- Cross-reference multiple sources when possible
- Clearly distinguish between facts and your interpretation
- Cite your sources by mentioning where you found the information
- If you're unsure, say so and suggest where to look
- Summarize findings clearly with key takeaways
- Save important findings to memory for future reference""",
        tool_ids=["web_search", "fetch_webpage", "calculator", "utc_now"],
        recommended_models=["gemma3:4b", "llama3.2:3b", "mistral:7b", "qwen2.5:7b"],
        default_settings={"temperature": 0.3, "max_tokens": 2048},
    ),
    SystemTemplate(
        id="coding_assistant",
        name="Coding Assistant",
        description="Expert programmer that reads, writes, and tests code in your workspace.",
        icon="code",
        category="coding",
        system_prompt="""You are an expert software engineer. You help the user write, debug, and improve code.

Guidelines:
- Read existing files before modifying them to understand context
- Write clean, well-structured code following the project's conventions
- Test your code by running it when possible
- Explain your changes and reasoning
- Use the workspace directory for all file operations
- Search for files to understand project structure before making changes
- When debugging, read error output carefully and fix the root cause""",
        tool_ids=["read_file", "write_file", "list_directory", "search_files", "run_python", "run_shell", "web_search"],
        recommended_models=["deepseek-coder-v2:16b", "codellama:7b", "qwen2.5-coder:7b", "gemma3:4b"],
        default_settings={"temperature": 0.1, "max_tokens": 4096},
    ),
    SystemTemplate(
        id="writing_assistant",
        name="Writing Assistant",
        description="Creative and technical writer, editor, and proofreader.",
        icon="edit_note",
        category="writing",
        system_prompt="""You are a skilled writer and editor. You help with creative writing, technical writing, editing, and proofreading.

Guidelines:
- Match the user's tone and style preferences
- For editing: preserve the author's voice while improving clarity
- For creative writing: be imaginative and engaging
- For technical writing: be precise, clear, and well-structured
- Offer multiple alternatives when the user asks for suggestions
- Use web search for fact-checking when writing about real topics
- Save drafts to files when working on longer pieces""",
        tool_ids=["read_file", "write_file", "web_search"],
        recommended_models=["gemma3:4b", "llama3.2:3b", "mistral:7b"],
        default_settings={"temperature": 0.7, "max_tokens": 4096},
    ),
    SystemTemplate(
        id="general_assistant",
        name="General Assistant",
        description="Helpful everyday assistant for any task — questions, planning, brainstorming, and more.",
        icon="smart_toy",
        category="general",
        system_prompt="""You are a helpful, knowledgeable assistant. You help with everyday tasks including answering questions, planning, brainstorming, calculations, and general problem-solving.

Guidelines:
- Be concise but thorough
- Search the web when you need current information
- Use the calculator for math instead of computing in your head
- Be honest about what you don't know
- Offer actionable suggestions
- Adapt your communication style to the user's needs""",
        tool_ids=["web_search", "calculator", "utc_now", "fetch_webpage"],
        recommended_models=["gemma3:4b", "llama3.2:3b", "mistral:7b", "qwen2.5:7b"],
        default_settings={"temperature": 0.5, "max_tokens": 2048},
    ),
    SystemTemplate(
        id="data_analyst",
        name="Data Analyst",
        description="Data analysis expert — works with CSV, JSON, SQL, and creates visualizations with Python.",
        icon="analytics",
        category="data",
        system_prompt="""You are a data analyst expert. You help users analyze data, generate insights, and create visualizations.

Guidelines:
- Read data files (CSV, JSON, etc.) from the workspace to understand their structure
- Write and run Python scripts for data analysis and visualization
- Use pandas, matplotlib, or other available libraries
- Always show your work — display the code and explain the output
- Save generated charts and reports to the workspace
- Provide clear summaries of key findings
- Suggest follow-up analyses when relevant""",
        tool_ids=["read_file", "write_file", "run_python", "calculator", "search_files", "list_directory"],
        recommended_models=["gemma3:4b", "qwen2.5:7b", "deepseek-coder-v2:16b"],
        default_settings={"temperature": 0.2, "max_tokens": 4096},
    ),
    SystemTemplate(
        id="image_creator",
        name="Image Creator",
        description="Creative image generation specialist — crafts detailed prompts for local diffusion models.",
        icon="palette",
        category="creative",
        system_prompt="""You are a creative image generation specialist. You help users create images using local diffusion models.

Guidelines:
- Craft detailed, descriptive prompts optimized for Stable Diffusion / Flux models
- Include style, lighting, composition, and quality descriptors in prompts
- Suggest appropriate settings (steps, guidance scale, dimensions) for the request
- Use web search to research art styles or reference images when needed
- Iterate on prompts based on user feedback
- Explain your prompt choices so users can learn prompt engineering""",
        tool_ids=["generate_image", "edit_image", "web_search"],
        recommended_models=["gemma3:4b", "llama3.2:3b", "mistral:7b"],
        default_settings={"temperature": 0.7, "max_tokens": 1024},
    ),
]

# Lookup by ID
TEMPLATES_BY_ID: dict[str, SystemTemplate] = {t.id: t for t in SYSTEM_TEMPLATES}


def list_templates() -> list[dict]:
    """Return all templates as serializable dicts."""
    return [
        {
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "icon": t.icon,
            "category": t.category,
            "tool_ids": t.tool_ids,
            "recommended_models": t.recommended_models,
            "default_settings": t.default_settings,
        }
        for t in SYSTEM_TEMPLATES
    ]


def get_template(template_id: str) -> SystemTemplate | None:
    return TEMPLATES_BY_ID.get(template_id)


# [IMPROVE-34] Shared helper used by BOTH the canonical
# ``POST /agents/from-template/{template_id}`` route and the deprecated
# ``POST /systems/deploy/{template_id}`` alias. Keeping the deploy logic
# here (instead of duplicating it across two routers) means the alias
# can never drift from the canonical endpoint — pin
# ``test_aliases_produce_equivalent_agent`` enforces this invariant.
#
# The doc rationale (``docs/features/05-systems.md:417-423``): the old
# URL is dishonest because the operation creates *an agent*, not a
# system. The new URL surfaces that truth. The alias stays for a
# release or two so existing Flutter clients (``systems_page.dart``)
# don't break the moment this lands.
def deploy_template_as_agent(
    template_id: str,
    body: dict | None,
    orchestrator,
) -> dict:
    """Instantiate ``template_id`` as a saved agent and return the
    canonical response payload.

    ``body`` accepts the same overrides the original endpoint took:
    ``name``, ``model_name``, ``provider``. Missing keys fall back to
    the template's defaults (first recommended model, ``ollama``
    provider, template id as agent name).

    Raises ``KeyError`` when ``template_id`` is unknown so the HTTP
    layer can map it to a 404 with a useful message — keeping HTTP
    concerns out of this pure helper.
    """
    if body is None:
        body = {}

    template = get_template(template_id)
    if template is None:
        raise KeyError(template_id)

    agent_name = body.get("name", template.id)
    model_name = body.get(
        "model_name",
        template.recommended_models[0]
        if template.recommended_models
        else "gemma3:4b",
    )
    provider = body.get("provider", "ollama")

    orchestrator.add_agent(
        name=agent_name,
        model_name=model_name,
        system_prompt=template.system_prompt,
        provider=provider,
        settings=template.default_settings,
        role="general",
    )
    if template.tool_ids:
        orchestrator.set_agent_tools(agent_name, template.tool_ids)

    # Late import — ``repositories.agents_repo`` pulls in DB code, and
    # ``system_templates`` is loaded by API-server bootstrap; keeping
    # this import lazy preserves the module's "data-only" character
    # at import time.
    from local_ai_platform.repositories.agents_repo import save_agent

    save_agent(agent_name, {
        "name": agent_name,
        "model_name": model_name,
        "system_prompt": template.system_prompt,
        "provider": provider,
        "settings": template.default_settings,
        "role": "general",
        "tool_ids": template.tool_ids,
        "template_id": template.id,
    })

    return {
        "status": "deployed",
        "agent": agent_name,
        "template": template.id,
        "tools": template.tool_ids,
    }
