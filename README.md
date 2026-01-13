# DeepResearch

DeepResearch is an agentic research pipeline designed to take a simple query and expand it into a detailed, multi-source research report. 

Built with the **OpenAI Agents SDK**, this project focuses on high-precision orchestration, ensuring that every step—from query clarification to final synthesis—is validated and reliable.

## The Foundation

DeepResearch is based on the principle of "asking before answering." The pipeline is structured around specialized agents that communicate through strict data contracts. This approach ensures that every piece of information moving through the system is typed, validated, and predictable.

### Current Progress
The core architectural patterns and shared data models are now established, providing the foundation for the researcher's lifecycle. 
- **Agent Orchestration**: Modular structure with isolated agent responsibilities.
- **Data Integrity**: Using Pydantic to enforce communication standards between agents.
- **Fail-Safe Patterns**: Implementation of the `AgentResult` wrapper to handle errors gracefully across agent boundaries.

## Exploration

To see the current data contracts in action or verify the foundation:

```bash
# Install the core dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the validation suite
pytest tests/test_schemas.py
```

## License
MIT License
