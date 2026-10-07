# Security Policy

This file is the **threat model**: what trelix exposes, what reaches an LLM, and what it
does not defend against.

**Do NOT open a public GitHub issue for a security vulnerability.** The reporting
policy — contact address, supported versions, response timeline, release-artifact
verification — is stated in [`.github/SECURITY.md`](.github/SECURITY.md) and only there.
That is the file GitHub resolves for the Security tab, because it takes `.github/` ahead
of the repository root, so it is where a reporter actually lands.

Keeping a second copy here is how the two files came to contradict each other. Through
v3.1.1 this file listed `1.x`/`0.7.x` as supported while `.github/SECURITY.md` listed
`2.7.x`/`2.6.x` — the shipping version was in neither — and the two promised different
acknowledgement windows (48 hours here, 2 business days there). Each fact below has one
home; the reporting facts live in the other file.

## Scope

trelix processes local repository contents and makes network calls to configured LLM/embedding providers. Security-sensitive areas include:

- **Credential handling** — API keys are read from environment variables and never logged or written to disk
- **MCP server** (trelix-mcp) — executes as a subprocess; only binds to stdio transport (not network)
- **File system access** — **symlinks are followed by default, including out of the
  repository.** A symlink under `repo_path` whose target lives elsewhere is indexed,
  and because `rel_path` is computed on the unresolved path it is then reported as
  though it sat inside the repo (a link `repo/linked_dir` to an out-of-tree
  directory yields `linked_dir/secret.py`, whose bytes are read from outside). Releases up to and including v3.1.0 documented the opposite;
  that claim was never true. Set `TRELIX_WALKER_FOLLOW_SYMLINKS=false` to confine the
  walk to `repo_path` by resolved path. It is opt-in because enabling it by default
  would silently drop files from any repository that symlinks to shared or vendored
  directories. Symlinks whose targets are *inside* the repo are still indexed either
  way — the setting is a boundary, not a blanket symlink filter. The GitHub App
  (`infra/github-app`) and this repository's `trelix-review.yml` workflow both set
  `TRELIX_WALKER_FOLLOW_SYMLINKS=false`, because they index pull requests written
  by outsiders; operators who index untrusted repositories should set it too.
- **Tree-sitter parsing** — parses user code with C-extension parsers; malformed inputs are caught and logged
- **Image and diagram connectors** (`trelix connector sync`) — do not use the symlink
  setting above: they always skip a symlink whose resolved target is outside the repo,
  so its contents are never sent to the configured LLM.
- **SQLite WAL-reset bug** — the index is a SQLite file in WAL mode, and SQLite 3.7.0
  through 3.51.2 can lose committed writes when two or more connections, in different
  threads or processes, write or checkpoint it at the same instant. The fix is in 3.51.3
  and was backported to 3.50.7 and 3.44.6. This is a risk to the integrity of the local
  index, not a disclosure risk. trelix has no SQLite of its own; it uses the build of the
  Python that runs it, including the Python inside a standalone binary or container image.
  When a writer opens the index on a build outside the fixed versions, trelix logs one
  warning per process (`wal_reset_risk` in `store/db.py`). The check goes by the
  upstream version number, so a distro build that backported the fix can still trigger it;
  the warning is a prompt to check, not a finding that the build is affected. Run one
  process that writes to an index at a time, or upgrade Python or SQLite. Read-only opens
  do not warn. See "trelix warned about SQLite WAL reset" in
  [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

### Embedding cache on disk (`TRELIX_EMBEDDING_CACHE_ENABLED`)

Off by default; nothing is written. When on, `Indexer` stores the float32 vector of every chunk
it embeds in `$XDG_CACHE_HOME/trelix/embeddings/<fingerprint>.db` (else
`~/.cache/trelix/embeddings/`, or `TRELIX_EMBEDDING_CACHE_DIR`, which must be an absolute path),
keyed by `sha256(chunk_text)`. The file holds no chunk text — hashes and vectors only — but a
vector is a lossy encoding of the text, and the hash lets anyone who can read the file test
whether a given text was ever indexed (membership inference). The cache therefore carries the
sensitivity of every index that wrote to it.

What the code does about it: the directory is created `0o700` and each file created `0o600`
(POSIX; Windows applies neither); the location comes only from the operator environment
(`TRELIX_EMBEDDING_CACHE_DIR`, `XDG_CACHE_HOME`, `HOME`), never from the repository or the
cwd; a relative directory, a `TRELIX_EMBEDDING_CACHE_MAX_MB` below 1, or an unusable directory
stops the run before any model is loaded rather than running uncached; indexed text is hashed and
its vector stored, never interpreted.

What it does not do: the hosted GitHub App (`infra/github-app`) forwards every `TRELIX_*` host
variable except `TRELIX_APP_*` and `TRELIX_GIT_TOKEN` to its `trelix index` child and passes
`HOME` and `XDG_CONFIG_HOME`, so the child also reads the operator env file. An App host that sets
`TRELIX_EMBEDDING_CACHE_ENABLED=true` in either place turns the cache on for every tenant's
index run, and every tenant then shares one `<fingerprint>.db` — a tenant who can read that file
can test whether another tenant's text was embedded. Never set it on a multi-tenant host. To
remove a cache, delete the `<fingerprint>.db` files (and any `-journal` sidecar) in that
directory.

### REST API — /graph/visualize output path constraint

The `output` query parameter on `GET /graph/visualize` is validated server-side:
- The resolved output path must be inside `<repo>/.trelix/`
- Paths outside this directory are rejected with HTTP 400
- This prevents arbitrary file writes to sensitive locations

### REST API — Host and Origin check (DNS rebinding, drive-by requests)

A loopback bind is a network boundary, not a browser boundary. With no credential
configured the API is open, and a web page can reach `127.0.0.1` in two ways: fire
`no-cors` GETs with side effects (`/ask` spends LLM credit; `/graph/visualize` writes
`<repo>/.trelix/graph.html`), or, after DNS rebinding (an attacker domain re-resolved to
`127.0.0.1`), become same-origin with the server and read `/search`, `/ask` and `/parse`.

- **What is checked.** `Host` must be present exactly once and its hostname (port ignored,
  case-insensitive) must be allowed. An `Origin` header, when present, must be an
  `http`/`https` origin with an allowed hostname (any port); `null` is refused. A
  `Sec-Fetch-Site: cross-site` request is refused. Anything else gets `403` with a fixed
  message that names the fix and never echoes the offending value; each rejection logs one
  warning. Requests with no `Origin` and no `Sec-Fetch-Site` (curl, scripts, the trelix
  clients) skip those two checks but still need an allowed `Host`: a client that reaches the
  server under another hostname gets `403` until that hostname is listed. There is no CORS
  support. A `no-cors` cross-site GET carries no `Origin`, so it is caught by
  `Sec-Fetch-Site`, which current Chrome, Firefox and Safari send; DNS rebinding is caught by
  the `Host` check in every browser.
- **When it is on.** `trelix serve` enables it when no `TRELIX_API_AUTH_TOKEN` and no OIDC
  is configured and the bind host is loopback (`127.0.0.0/8`, `::1`, `localhost`); the
  allowed names are `localhost`, `127.0.0.1`, `::1` and the bind host. It stays off when a
  credential is configured (a hostile page has no key) and for a non-loopback bind, unless
  `TRELIX_API_ALLOWED_HOSTS` lists hostnames, which turns it on for any bind. `GET /health`
  is exempt so container and Kubernetes probes keep working. A container binds `0.0.0.0`, so
  the documented `docker run -p 127.0.0.1:8765:8765 ... serve /repo --host 0.0.0.0` shape
  needs `-e 'TRELIX_API_ALLOWED_HOSTS=localhost,127.0.0.1,[::1]'` (keep the quotes: zsh
  treats an unquoted `[::1]` as a glob). `docker-compose.yml` sets it for its
  loopback-published port; the Helm chart leaves it off, so add the hostnames your Service
  and Ingress use under `extraEnv`.
- **Adding a hostname.** Set `TRELIX_API_ALLOWED_HOSTS=dev.example,other.example`
  (comma-separated bare hostnames; ports, brackets and case are ignored; URLs with a scheme
  and wildcards such as `*.example.com` are not supported and are dropped with a warning).
  Behind a reverse proxy the `Host` header must reach trelix with a listed name.
- **Disabling it.** `TRELIX_API_ALLOWED_HOSTS=*` switches the check off in every case.
- **What it is not.** It does not replace `TRELIX_API_AUTH_TOKEN`: it does nothing against a
  non-browser client that can reach the port and sets any `Host` it likes. See
  [docs/CONFIGURATION.md](docs/CONFIGURATION.md#rest-api).
- **Known limits.** A browser that sends no `Sec-Fetch-Site` (older Safari and other old
  browsers) does not stop a `no-cors` cross-site GET to a loopback bind; DNS rebinding is
  still caught there by the `Host` check. Any other web app served from a loopback address,
  on any port, is an allowed origin. A browser request to `http://0.0.0.0:<port>` carries
  that `Host` and gets `403`.

### MCP federation tools — config_path confinement (v2.8.1+)

The `config_path` parameter accepted by `federation_list_repos`,
`federation_add_repo`, `federation_remove_repo`, and `federation_search_all`
is validated server-side:
- The resolved path must be inside `~/.config/trelix/` or `<mcp-server-cwd>/.trelix/`
- Paths outside these roots are rejected with an `{"error": ...}` response
  (never raises — matches the "never raise" convention of every other
  federation tool)
- Like the `/graph/visualize` check above (`api/app.py:494`), this uses
  `Path.is_relative_to()` on the resolved path rather than a string-prefix
  comparison, which avoids a sibling-directory bypass (e.g.
  `~/.config/trelixevil/` would incorrectly pass a naive
  `startswith("~/.config/trelix")` check). Earlier revisions of this document
  described the two checks as differing on this point; they do not
- Also caps the federation registry at `TRELIX_FEDERATION_MAX_REPOS`
  (default 50) entries and the number of repos actually queried per
  `federation_search_all` call, preventing a scripted/adversarial MCP
  client from growing the registry or fan-out unboundedly

### Claude Code plugin — what it runs, reads and writes (`plugins/trelix/`)

The plugin (`.claude-plugin/marketplace.json`, `plugins/trelix/`) runs exactly two
commands, both listed with their purpose in `plugins/trelix/README.md`:

- `uvx --from trelix-mcp==<pin> trelix-mcp`: the MCP server, the same `trelix-mcp`
  that `claude mcp add trelix -- trelix-mcp` registers, at the pinned published
  release. Everything above about the server applies to it unchanged.
- `uv run --no-project --script "${CLAUDE_PLUGIN_ROOT}/scripts/session_start.py"`,
  the SessionStart hook (`startup`, `resume`, `clear`; 15 s timeout). The script is
  standard-library Python (a test pins its import roots and that the name `urlopen`
  does not appear in it), makes no network call, reads `CLAUDE_PROJECT_DIR` or the
  hook's stdin `cwd`, and opens only `<project>/.trelix/index.db`, through a
  `mode=ro` SQLite URI: a missing index is not created and a write would fail
  (SQLite may create empty `-wal`/`-shm` sidecars beside an existing WAL index, as
  any read-only open does; when `.trelix/` is not writable and they are absent it
  cannot open a WAL index at all, and the hook prints nothing). When the index
  records the commit it was built from (checked against `^[0-9a-f]{7,64}$` before
  it enters argv) the script runs one child,
  `git -C <project> rev-list --count <commit>..HEAD`, with
  `GIT_TERMINAL_PROMPT=0`, `GIT_OPTIONAL_LOCKS=0` and a 5 s timeout. It prints at
  most 1,000 characters (the project path, row counts, the build time, 12
  characters of the commit, the distance, the embedder provider; never a model
  name, never a key) and exits 0 on every path; on an error it prints nothing and
  writes `trelix session_start: skipped (<exception type>)` to stderr, which Claude
  Code keeps in its debug log only.

The plugin files carry no secrets and set no environment variables (`.mcp.json` has
no `env`; provider keys reach the server from the user's shell or
`~/.config/trelix/env`). Hook commands run with the user's permissions and inherit
Claude Code's environment, outside any sandbox (a paraphrase of the hooks
documentation), on the repository the user opened and Claude Code's own trust dialog
accepted; beyond the commit-shape check the hook is not hardened against a hostile
repository.

The skill's `allowed-tools` pre-approves `search_code`, `get_symbol` and
`blast_radius`: the three tools annotated `readOnlyHint` on this repository's
`develop` branch (`packages/trelix-mcp/tests/test_tool_readonly.py`). On the pinned
3.4.3 a call to any of them against an unindexed repository creates an empty,
gitignored `.trelix/index.db` and answers empty; nothing else is written. The grant
is kept because the SessionStart line and the skill forbid searching before
indexing, and it lasts one turn (Claude Code clears it at the next message); the
pin bump to the next release makes the statement exact.

## Out of Scope

- Vulnerabilities in third-party dependencies (report to upstream)
- Denial-of-service via extremely large repositories (use `--limit` flags)
- Issues requiring physical access to the machine

## Prompt Injection via Indexed Content

**Status: documented, not mitigated.** trelix ships no defence against prompt
injection. This section exists so you can decide what to index and what to scope
a token to — not because a control was added. Everything below is a byte-level
statement about which text reaches a model, established by reading the cited code
paths. Nothing below is a claim about how a model responds to that text; see
"What is unmeasured" at the end of this section.

### Attacker and attacker-controlled surface

- **The attacker is anyone who can land text in a file trelix indexes.** Repo
  write access is not required — for the review path below, an unmerged pull
  request is enough.
- **Indexed content is re-emitted verbatim.** A symbol's body is copied into its
  chunk text (`indexing/chunker.py:149`), and the chunk text is copied into the
  assembled LLM context (`retrieval/assembler.py:334`). Nothing transforms it in
  between.
- **Prose-carrying formats are indexed by default, not just code.** Markdown,
  HTML, YAML, TOML and JSON are all in the default `languages` list
  (`core/config.py:30-52`). `.md`/`.mdx` map to `Language.MARKDOWN`
  (`indexing/walker.py:47-48`) and are parsed into heading-based sections
  (`indexing/parser/registry.py:132-136`). Free-form English in a README is
  first-class indexed content, not an edge case.
- **Assume attacker influence whenever the repository** accepts pull requests or
  patches from outside your trust boundary; holds test fixtures, recorded HTTP
  responses, or sample payloads; or vendors third-party source. `node_modules`,
  `vendor` and `Pods` are ignored by default (`core/config.py:62-95`) — but
  `third_party/`, `external/`, git submodules and a checked-in SDK are not.

Exposure by default vs opt-in:

| Path | Default | What reaches a model |
| ---- | ------- | -------------------- |
| `trelix index` | on | with the default `local` embedder (`EmbedderConfig.provider` in `core/config.py`), nothing leaves the machine; with any remote embedder, every chunk's text is sent to the embedding model (`indexing/indexer.py:959`, `:1021`) |
| `trelix ask`, `GET /ask` | on | the assembled retrieval context (below). With `TRELIX_RETRIEVAL_CITATIONS=true` its blocks carry `[C#]` tags; the model's markers are read back as digits only (`retrieval/citations.py`), and the verifier opens only files the index named, under the repository root, for a newline count |
| index-time file summaries | off — `TRELIX_FILE_SUMMARIES_ENABLED` (`core/config.py:1234-1237`) | file path, language, and top symbol signatures truncated to 80 chars (`indexing/file_summarizer.py:85-91`) |
| agentic loop | off — `TRELIX_RETRIEVAL_AGENTIC` (`core/config.py:656-659`), or `--agentic`/`--session` (`cli/main.py:422-424`). **The MCP `ask_agent` tool ignores this** and forces the loop on unconditionally (`trelix_mcp/server.py:762`); it does require an LLM to be configured. | retrieval context plus prior-turn observations |
| `trelix review` | opt-in command | diff hunk text plus retrieved context |
| `trelix review --post-comments` | off (`cli/main.py:1540-1546`) | as above, and model output leaves the machine |
| `trelix-mcp` tools and resources | on, once the server is registered with a host | verbatim symbol bodies — `search_code` 800 chars, `get_symbol` and the `trelix://…/symbol` resource the **whole** body |
| `trelix-mcp` `federation_search_all` | on, once repos are registered | verbatim bodies (800 chars) from **every registered repository**. It takes no `repo_path` argument — scope comes from the registry file, not the caller — so a host agent that named no repository can receive content from repositories it never opened. `trelix search-all` emits no body bytes and no REST equivalent exists, so this is the only cross-repo content egress in the product. |

The `trelix-mcp` row deserves separate emphasis, because its sink is the widest
in this table. The consumer of an MCP tool result is a host agent — Claude Code
and its peers — which typically holds shell and file-write tools of its own, a
strictly broader capability than the single authenticated GitHub comment write
described below. To be precise about what trelix itself directs: its MCP prompt
templates instruct the host agent to call trelix's own read-only tools
(`search_code`, `get_symbol`, `blast_radius` — `trelix_mcp/prompts.py`), not the
host's shell or file-write tools. The wider capability belongs to the consumer,
not to anything trelix asks it to do.

### What reaches a model, and in what form

1. **Assembled retrieval context, in the same message as your question.**
   `_USER_TEMPLATE` (`retrieval/synthesizer.py:88-95`) interpolates
   `context_text` and `query` into a single `role="user"` message
   (`retrieval/synthesizer.py:277-280`, `:325-328`). `context_text` is the output
   of `ContextAssembler._format_context` (`retrieval/assembler.py:291-337`),
   which emits each result's chunk text verbatim under a `=== <path> ===` /
   `[Lines a-b] <qualified_name>` header. `GET /ask` runs the same two steps
   (`api/app.py:431-449`).
2. **Agent-loop observations, carried across turns and across processes.** The
   `retrieve` action places up to 300 characters of each matching symbol body in
   the observation (`agent/loop.py:238`); `get_symbol` places the whole body in
   (`agent/loop.py:275`). `TurnHistory.to_text()` renders each observation back
   into the next turn's prompt, truncated to 500 characters
   (`agent/history.py:87-99`), and that string is concatenated into the user
   message at `agent/loop.py:168-175`. Turns are persisted to SQLite
   (`agent/loop.py:146`) and replayed into the prompt when a session is resumed
   with `--session` (`agent/loop.py:104`, `:112`) — text read in one invocation
   can therefore re-enter a prompt in a later one. This is about the *text* the
   loop forwards; the loop's *action* confinement is unchanged and is described
   under "Agentic Loop Security" in the v2.2.0 notes below.
3. **Diff text plus retrieved context, in the review path.** Each hunk's removed
   and added lines are formatted into a user message, followed by up to 3,000
   characters of retrieved context (`review/reviewer.py:137`, `:147-154`), and
   sent with a review system prompt (`review/reviewer.py:156-161`). Under `--pr`
   the diff is the pull-request author's patch, fetched from the GitHub API
   (`review/github.py:102`) and reassembled into a unified diff at
   `cli/main.py:1594-1605`.
4. **Indexed text also reaches *you* with no model in the loop.** When the agent
   loop hits its turn cap it returns the first three successful observations as
   the answer (`AgentLoop._fallback_answer()` in `agent/loop.py`). With a `local` embedder provider
   (and none of `TRELIX_RETRIEVAL_FLARE=true`, `TRELIX_RETRIEVAL_AGENTIC=true`,
   `--agentic` or `--session`), `trelix ask` prints the assembled context and returns
   before any synthesis call (`ask()` in `cli/main.py`). Terminal rendering is escaped
   for display only (see the Rich-markup notes in `cli/main.py`) — escaping prevents
   markup errors, it is not sanitization.

- **The index outlives its source, and re-indexing does not reclaim it.** A symbol's
  body is stored in SQLite, so it is still returned after the file is deleted from
  disk — verified by indexing a file, unlinking it, and getting the verbatim body
  back. Any reasoning of the form "the consumer could have read that file itself
  anyway" therefore does not hold for content that has since been removed.

  **Re-indexing is not a remedy.** `trelix index` walks the filesystem and upserts
  what it finds; it never reconciles the DB's file list against the walk, so a row
  whose file is gone is never reaped. Verified for the incremental default, for
  `incremental=false`, and for two consecutive runs — the deleted file's body
  survived all three. The only paths that delete a vanished file's symbols are the
  live watchers (`indexing/watcher.py:240`, `indexing/multi_watcher.py:144`), which
  act on filesystem delete events as they happen and cannot recover a deletion that
  occurred while no watcher was running.

  To actually purge content, delete `<repo>/.trelix/index.db` and index again
  (verified: the symbol and its body are then absent).

### What trelix does not do about it

Stated as plainly as any benefit in this document:

- **No sanitization.** Indexed content is never filtered, neutralized, or
  rewritten before it enters a prompt. There is no character allowlist, no
  instruction-stripping, and no rejection of anomalous content. The size caps
  cited above exist for token budget, not for safety.
- **No instruction/data separation.** Retrieved content and the user's question
  share one `role="user"` message on every retrieval, agent, and review path
  above — repository text and the instruction to act on it are the same
  undelimited string by the time the request is built. Where a system prompt
  exists it carries trelix's own instructions; it does not fence the retrieved
  text. Code fences in these prompts are a rendering guarantee, not a trust
  boundary. Their length is derived from the payload (`llm/prompt.py`,
  `fence_for`), so a payload containing its own ``` can no longer close the block
  early — but a fence only marks content, it does not constrain what a model does
  with it. That change, and the terminal-escaping change referenced above, are
  correctness fixes to rendering. **Neither is an anti-injection control, and
  neither was measured against a model.**
- **No detection.** trelix does not scan indexed content, prompts, or model
  output for injection attempts, and emits no log line, metric, or exit code when
  content looks anomalous.
- **No provenance tracking.** Once text is in the index, nothing records whether
  it came from a first-party file or a contributed/vendored one, so nothing
  downstream can treat the two differently.

### Highest-severity capability chain: `trelix review --pr --post-comments`

Each link read from the code:

1. `--pr` requires `GITHUB_TOKEN` (`cli/main.py:1565-1570`) and fetches the PR's
   patches (`review/github.py:102`).
2. Those patches — authored by the PR author, who need not be trusted — are
   placed in the LLM user message (`review/reviewer.py:154-161`).
3. The model's returned text is parsed into `ReviewComment.comment`
   (`review/reviewer.py:180`).
4. With `--post-comments`, each `comment` becomes the `body` of an inline GitHub
   review comment (`cli/main.py:1670-1678`), posted with that same token
   (`cli/main.py:1679-1686`, `review/github.py:146-207`).

So untrusted repository text sits at one end of a chain whose other end is an
authenticated write to GitHub. Two limits bound the chain rather than break it,
and are worth stating precisely:

- trelix never passes `event=`, so the review is created as the default
  `"COMMENT"` (`review/github.py:154`) — this path cannot approve a PR.
- The only field derived from model output is the comment `body`; `path` and
  `line` come from the parsed diff (`cli/main.py:1671-1673`).

**No exploit was attempted.** What is asserted here is capability — that the code
connects untrusted input to a write-scoped credential — not that any particular
model would traverse it.

### What is unmeasured

The analysis behind this section made **zero LLM calls**. Consequently:

- Whether any model acts on instructions embedded in indexed content is
  **unmeasured**, for every provider and every prompt above.
- Which path is most susceptible is **unmeasured** — do not read the ordering
  above as a ranking of exploitability.
- No mitigation is recommended here because none was evaluated. Any candidate
  defence would need model-in-the-loop measurement that has not been done.

### External evidence this section's severity is not hypothetical

Independent academic research (2026) quantifies the two attack classes this
section describes structurally:

- **MCP tool poisoning** (a malicious/compromised tool description
  manipulating an LLM client) was demonstrated at scale against 45 real,
  live production MCP servers — 36.5% average attack-success rate across 20
  LLM-agent configurations, peaking at 72.8%; even the best-refusing model
  tested rejected fewer than 3% of attempts. Source: MCPTox
  (arXiv:2508.14925).
- **Indirect prompt injection via retrieved/indexed content** — the class
  this section's "Assembled retrieval context" and MCP-tool rows describe —
  was shown to succeed 80–93% of the time from poisoning as little as 0.7%
  of a code-retrieval corpus, via ordinary top-ranked retrieval (no
  privileged access required). Defenses evaluated in the same research
  reduced but did not eliminate this: 40–71% residual attack success even
  with mitigations applied. Sources: CodePoisonRAG and "Beyond the Payload"
  (both Sept 2026 preprints; see `docs/reports/post-v3.3.0-research-sweep-2026-09-16.md`
  Tier 4 for full citations).

Neither paper evaluates trelix specifically — both describe the general
attack class an MCP server and a retrieval-augmented agent loop are exposed
to. Nothing above changes the "documented, not mitigated" status: no
mitigation evaluated in this research achieves reliable protection, so none
is claimed here either.

### Practical guidance

Before indexing a repository you do not fully trust:

- Treat every prose file in it as prompt content, not just the code. A single
  `trelix ask` sends the retrieval-selected subset that fits the assembler's
  token budget, not the whole repository — but across enough queries any indexed
  chunk can reach a prompt. Note that on the *default* configuration `trelix ask`
  makes no synthesis call (the `local` embedder returns the assembled context
  directly; only the question text goes to a configured chat model, for query planning);
  the exposure below begins once an LLM-backed path is configured.
- Leave the agentic loop off (`TRELIX_RETRIEVAL_AGENTIC` is already `false`) — it
  is the only path that re-feeds prior observations into a later prompt. **This
  does not cover the MCP `ask_agent` tool**, which forces the loop on regardless
  of the variable; if a host has trelix-mcp registered, do not rely on the flag
  alone. The other multi-round path, FLARE, re-retrieves with a fixed query suffix
  and carries no prior content forward (`retrieval/flare.py:101-103`).
- Narrow what gets indexed: drop prose formats you do not need from
  `WalkerConfig.languages` and add vendored/fixture directories to
  `WalkerConfig.extra_ignore_dirs` (`core/config.py:30`, `:62`; env prefix
  `TRELIX_WALKER_`, `core/config.py:28`).
- Confine the walk to the repository with `TRELIX_WALKER_FOLLOW_SYMLINKS=false` if
  the tree contains symlinks you did not place there. By default a symlink out of
  the repository is indexed and reported under an in-repo path; see "File system
  access" under Scope above.
- For retrieval-only use, run with a `local` embedder provider — `trelix ask`
  then skips synthesis and prints the context (unless `TRELIX_RETRIEVAL_FLARE=true`,
  `TRELIX_RETRIEVAL_AGENTIC=true`, `--agentic` or `--session` is used; see `ask()` in
  `cli/main.py`), and indexing sends no chunk text to a remote embedding model.
  Retrieval's query planner is separate: while a chat credential or `TRELIX_LLM_BASE_URL` is
  set it still sends the question text to the chat model, at most once per query per process (the in-memory
  plan cache does not persist between CLI runs; `docs/FAQ.md` explains how to switch that
  off).
  `local` is already the default (`EmbedderConfig.provider` in `core/config.py`), but a
  `.env` or an exported
  `TRELIX_EMBEDDER_PROVIDER` can silently change it, so check the resolved value
  rather than assuming.

Scoping a review token:

- Omit `--post-comments` and the GitHub path is read-only: the only write is
  `post_review` (`review/github.py:146`), reached solely from the
  `--post-comments` branch (`cli/main.py:1679`).
- When you do post, use a fine-grained token limited to the single target
  repository, with the scopes named in the client docstring
  (`review/github.py:54-57` — `pull_requests:write`, `contents:read`). Do not
  reuse a classic `repo`-scope token: `repo` grants code write across every
  repository the token can reach, which is far wider than this feature needs.
- Prefer a short-lived CI job token over a long-lived personal token, and treat
  posted comments as untrusted model output when reading them.

## v2.1.0 Security Notes

### Query Telemetry (`telemetry_enabled`)

- **Storage**: Telemetry data (query text, intent, elapsed_ms, result_count) is stored locally in `.trelix/index.db` only
- **No external transmission**: All telemetry is SQLite-only; no data leaves the machine or contacts external services
- **Sensitive query strings**: If your codebase contains secrets in symbol names or comments, those strings may appear in query telemetry logs
- **Mitigation**: Disable telemetry via environment variable `TRELIX_TELEMETRY_ENABLED=false` (default is enabled in v2.1.0)

### Eval Golden Files (`trelix eval --golden`)

- **File content**: Golden JSONL files may contain internal query strings and evaluation assertions
- **Treat as internal documentation**: Golden files contain test data and should not be committed to public repositories
- **Recommended storage**: Store golden files in `.trelix/` directory (already gitignored) rather than repo root to prevent accidental disclosure

### HyDE Synthetic Snippets

- **Transient generation**: HyDE generates synthetic code snippets via LLM calls; these snippets are never persisted to disk
- **LLM exposure**: Only the normal LLM provider data-transmission path applies (i.e., no additional sensitive data is sent beyond what standard similarity search already sends to your LLM provider)
- **No local storage**: Synthetic snippets are embedded transiently in memory for ranking; they are discarded after the search completes

## v2.2.0 Security Notes

### Agentic Loop Security
- AgentLoop executes `retrieve`, `grep`, and `get_symbol` actions only — no code execution, no shell commands
- All actions read from the indexed SQLite DB; no external network calls during agent turns
- `max_results` on grep is capped at 50 (enforced in loop.py._do_grep)
- Disable: `TRELIX_RETRIEVAL_AGENTIC=false` (default)

### Taint Analysis Security
- TaintAnalyzer runs the Semgrep CLI via subprocess with a 120-second timeout
- Semgrep operates on local files only; no data leaves the machine
- Rule files: use `--rules <path>` to restrict to known-good rules; default uses Semgrep registry (requires internet)
- Results stored in `taint_flows` SQLite table (local, not transmitted)
- Disable: `TRELIX_PARSER_TAINT=false` (default)

### Sparse Embeddings
- SPLADE-Code model weights loaded from HuggingFace at first use (internet required once, cached locally)
- Sparse vectors stored in `sparse_embeddings` SQLite table (local only)
- No query data sent externally when using local inference
