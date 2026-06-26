# trelix-mcp

<!-- mcp-name: io.github.sairam0424/trelix -->

MCP server for [trelix](https://github.com/sairam0424/trelix) — semantic code search for Claude Code, Cursor, Windsurf, and Continue.dev.

## Install

```bash
pip install trelix-mcp
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

## Tools

| Tool | Description |
|------|-------------|
| `search_code` | Semantic hybrid search over an indexed codebase |
| `index_codebase` | Index a repository so it can be searched |
| `get_symbol` | Look up a symbol by qualified name |
| `blast_radius` | Find all files that depend on a symbol |
