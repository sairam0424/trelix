# trelix-mcp

<!-- mcp-name: io.github.sairam0424/trelix -->

MCP server for [trelix](https://github.com/sairam0424/trelix) v0.7.0 — semantic code search for Claude Code, Cursor, Windsurf, and Continue.dev.

## Install

```bash
pip install trelix-mcp
```

To use Bedrock embeddings or synthesis (no extra API key beyond AWS credentials):

```bash
pip install "trelix-mcp" "trelix[bedrock]"
```

Other optional LLM provider extras:

```bash
pip install "trelix-mcp" "trelix[anthropic]"   # Anthropic Claude direct
pip install "trelix-mcp" "trelix[vertex]"       # Google Vertex AI / Gemini
pip install "trelix-mcp" "trelix[litellm]"      # 100+ providers via LiteLLM
pip install "trelix-mcp" "trelix[llm-all]"      # all LLM providers
```

## Usage

### Claude Code

```bash
claude mcp add trelix -- trelix-mcp
```

### Cursor (`~/.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "trelix": {
      "command": "trelix-mcp",
      "args": []
    }
  }
}
```

### Continue.dev (`.continue/config.json`)

```json
{
  "mcpServers": [
    {
      "name": "trelix",
      "command": "trelix-mcp",
      "args": []
    }
  ]
}
```

## Configuration

Set environment variables before starting the MCP server. All variables are optional — defaults work out of the box with the `local` embedding provider and `openai` chat provider.

### Embedding provider

```bash
# Local sentence-transformers — no API key (default)
TRELIX_EMBEDDER_PROVIDER=local

# Azure OpenAI embeddings
TRELIX_EMBEDDER_PROVIDER=azure
AZURE_API_KEY=...
AZURE_ENDPOINT=https://<resource>.openai.azure.com/

# Voyage AI — best API-based code embeddings (CoIR 56.26)
TRELIX_EMBEDDER_PROVIDER=voyage
VOYAGE_API_KEY=...

# AWS Bedrock Cohere — strong code retrieval, no extra key beyond AWS creds
TRELIX_EMBEDDER_PROVIDER=bedrock-cohere
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# AWS Bedrock Titan v2 — configurable 256/512/1024 dims
TRELIX_EMBEDDER_PROVIDER=bedrock-titan
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

### Chat / synthesis provider (used by `index_codebase` contextual chunking and synthesis)

```bash
# OpenAI (default)
TRELIX_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Azure GPT-4o
TRELIX_LLM_PROVIDER=azure
AZURE_API_KEY=...
AZURE_ENDPOINT=https://<resource>.openai.azure.com/

# AWS Bedrock — Claude Sonnet 4.6 default with auto-fallback to Haiku
TRELIX_LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
# Optional overrides:
TRELIX_LLM_BEDROCK_PRIMARY_MODEL=us.anthropic.claude-sonnet-4-6
TRELIX_LLM_BEDROCK_FALLBACK_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0

# Anthropic direct
TRELIX_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Vertex AI / Gemini
TRELIX_LLM_PROVIDER=vertex
GOOGLE_CLOUD_PROJECT=my-project
GOOGLE_CLOUD_LOCATION=us-central1

# LiteLLM — 100+ providers
TRELIX_LLM_PROVIDER=litellm
TRELIX_LLM_MODEL=bedrock/claude-3-5-sonnet
```

## Tools

| Tool | Description |
|------|-------------|
| `search_code` | Semantic hybrid search over an indexed codebase |
| `index_codebase` | Index a repository so it can be searched |
| `get_symbol` | Look up a symbol by qualified name |
| `blast_radius` | Find all files that depend on a symbol |
