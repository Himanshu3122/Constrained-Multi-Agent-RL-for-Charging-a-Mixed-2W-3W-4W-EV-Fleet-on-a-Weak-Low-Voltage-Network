# Security

## Scope and posture

This is research software. It has **not** been security-reviewed and must not control physical chargers or grid equipment without independent engineering and security validation.

Known limitations of the API (`solace.api`):

- no authentication or authorization;
- no rate limiting;
- no persistent or tamper-evident audit log;
- the intent layer's resistance to prompt injection is a design argument (the LLM can only emit a typed constraint that is then validated, feasibility-checked and projected), **not** a tested property;
- loading an LLM (`SOLACE_INTENT_PARSER=hf`) downloads third-party model code and weights.

Safeguards that are implemented: input validation with Pydantic (sizes, ranges, network consistency), no shell execution, no secrets in the repository (see `.env.example`), and a non-root user in the Docker image.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the maintainers (repository contact) rather than in a public issue. Include steps to reproduce and the affected version or commit.
