# trelix PR review — GitHub App and Actions workflow

Automatic PR review as a GitHub Check run with inline annotations on every
pull request. Two integration paths exist, covering the same review
capability:

| | GitHub Actions workflow | GitHub App (this directory) |
|---|---|---|
| Setup | Merge one YAML file into the repo | Install the App — no workflow file needed |
| Trust | Repo's own `GITHUB_TOKEN`, no third party | Grants a third-party App `pull_requests`/`checks`/`contents` access |
| Where reviews run | The installing repo's own Actions runners | This standalone service |
| Status | ✅ Shipped | ✅ Installable and hardened (see "Status" below) |

Pick the Actions workflow if you'd rather not install a third-party App.
Pick the App for zero-setup installability across many repos.

## Option 1: GitHub Actions workflow (shipped)

1. The workflow at `.github/workflows/trelix-review.yml` triggers on every
   `pull_request` event (`opened`/`synchronize`/`reopened`)
2. trelix indexes the repository (local embedder — no API key needed)
3. `trelix review --pr owner/repo#N --json` fetches the diff and reviews
   each changed hunk
4. Findings are posted as GitHub Check annotations with file + line
   references

### Quick setup

The workflow uses `GITHUB_TOKEN` (auto-provided by Actions) — no App
registration required.

1. Merge the PR that adds `.github/workflows/trelix-review.yml` to your repo
2. On the next pull request, the `trelix Code Review` check runs
   automatically (it is attached to the pull request's head commit, so it shows in the
   pull request's Checks tab)

### Required: an LLM provider

`trelix review` needs an LLM provider to produce anything (see the behavior
notes below). Set one of these repository secrets:

| Secret | Provider |
|--------|---------|
| `OPENAI_API_KEY` | OpenAI GPT-4o |
| `ANTHROPIC_API_KEY` | Anthropic Claude |
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | AWS Bedrock |

Add under **Settings → Secrets and variables → Actions → Repository
secrets**, then update the workflow's `env:` block to pass the key.

### Behavior notes

- The index step is tolerant — if indexing fails (network-restricted CI,
  OOM), the workflow prints a warning and still runs the review against a
  partial or empty index rather than blocking the PR
- Review findings are capped at 50 annotations per PR (GitHub API limit)
- Works on private repos — `GITHUB_TOKEN` scopes are sufficient
- `trelix review` needs a working LLM provider: it has no structural-only
  fallback. Without one it exits with code 3, and so does a review in which no
  hunk got a usable review (every reply was cut off, refused, filtered or not a
  review, or the call failed) and none kept a finding. The Check run is posted
  as **neutral** ("trelix review did not run"), never as "found 0 issue(s)". The
  workflow and the App both post on the pull request head commit
- A review that covered only part of the diff exits with code 4 after printing
  the findings it has (`TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION`, default `0`, is
  the share of unreviewed hunks it tolerates; at `1` it exits 0). The workflow
  and the App then post a **"trelix review incomplete"** Check: the findings as
  annotations, the counts ("3 of 5 hunks were reviewed and 2 were not") and the
  first ten unreviewed hunks as `` `file:line` (status) ``, with "and N more" for
  the rest. The conclusion is **neutral**, or **failure** if any finding is an
  `ERROR` (judged by every finding, not only the 50 that get an annotation);
  never success. The summary carries no model text: only numbers, the fixed
  statuses (`truncated`, `refused`, `parse_failed`, `error`) and the file names,
  which are pull-request text and are shown one to a line inside a code span,
  cut at 100 characters, without hidden characters or backticks (in the App they
  pass through the sanitiser as well). The App's rendering is the lossier of the
  two: it counts UTF-16 code units where the workflow counts characters, and its
  sanitiser also turns `@`, `<` and `>` into fullwidth look-alikes and breaks up
  link syntax, so a path such as `node_modules/@scope/x.js` reads slightly
  differently in the two Checks
- The counts and the list come from the record `trelix review` writes to
  `TRELIX_REVIEW_OUTCOME_FILE`. The workflow sets it to a fixed path
  (`/tmp/trelix-review-outcome.json`, removed before the review runs); the App
  gives each review its own private directory outside the checkout
  (`trelix-review-outcome-*` in the OS temp directory, removed when the review
  ends). Either reads the record as untrusted input: a record that is missing,
  not a regular file, over 1 MiB, not JSON, of another schema or exit code, or
  whose counts do not add up is **unknown**, and the Check says that how much was
  left unreviewed is unknown instead of listing hunks. It is never read as
  "nothing was left out". Findings that cannot be read from stdout (missing, not
  JSON, not an array, or an array holding something that is not an object) are
  reported the same way, as unread rather than absent, and after a clean exit 0
  too: the Check is **neutral** ("trelix review did not complete"), never
  "found 0 issue(s)", in the workflow and in the App alike

### Permissions required

```yaml
permissions:
  pull-requests: write   # post PR comments
  checks: write           # post Check annotations
  contents: read          # checkout code
```

These are declared in the workflow YAML and require no manual
configuration.

---

## Option 2: standalone GitHub App (this directory's TypeScript service)

A webhook-driven App: install it on a repo and PR reviews happen
automatically, with **zero workflow YAML required in the installing
repository**. Its webhook handler runs `trelix review --pr` directly and
posts Check annotations via Octokit — no dependency on the installing repo
having any Actions workflow at all. This is the App's whole reason to
exist over the Actions workflow above: genuine zero-setup installability
across many repos, at the cost of installing a third-party App with real
permissions.

### Architecture

```
GitHub -- pull_request webhook -->  this service (Express)
                                       |
                                       v
                              trelix review --pr ... --json
                                       |
                                       v
                              GitHub Checks API (annotations)
```

### Status: installable, hardened, deployable on demand

- ✅ **Signature verification.** `src/webhook.ts` verifies
  `X-Hub-Signature-256` (HMAC-SHA256 over the raw request body, keyed by
  the webhook secret) via `@octokit/webhooks-methods`'s `verify()`, which
  compares using `crypto.timingSafeEqual` — not a naive string compare.
  Requests with a missing, wrong-secret, or body-tampered-after-signing
  signature are rejected with `401` before the route handler ever sees
  the payload.
- ✅ **Installation-token minting, one token per purpose.** `src/auth.ts`'s
  `getInstallationToken` uses `@octokit/auth-app` (App-ID + private-key JWT
  signing -> installation-token exchange), with one `AuthInterface` reused
  per `AppConfig` so the library's own expiry-aware cache actually has a
  chance to hit across calls instead of re-minting on every request. Every
  token is limited to the one repository the pull request is in and to the
  permissions of one job: see "Token scopes" below.
- ✅ **Clone-on-demand.** `src/repo-checkout.ts`'s `checkoutPullRequest`
  clones the *actual* PR being reviewed into a fresh, per-request temp
  workspace (`mkdtemp`, cleaned up in a `finally` block) using the
  `checkout` installation token — there is no static, hardcoded repo path
  anymore.
  Fetches `refs/pull/<n>/head` against the base repo (never
  `--branch=<head.ref>`, which only exists on a fork's own repo for an
  external-contributor PR) and authenticates via a per-workspace
  `GIT_ASKPASS` script, kept outside the checkout, that reads the token
  from an env var — the token is never embedded in the remote URL,
  written to `.git/config`, or passed as a subprocess argument.
  Deliberately never passes
  `--recurse-submodules`: this service clones PRs from arbitrary external
  contributors, and an attacker-controlled `.gitmodules` is a real risk
  that diff-level review doesn't need to take on.
- ✅ **Check-annotation posting.** `runReview` mints the three tokens, clones
  and indexes the PR's actual head, runs the CLI review against that clone,
  and posts a completed Check run with inline annotations via
  `octokit.rest.checks.create`, on the head commit of the webhook delivery.
  Everything in it is sanitised first (see "Everything the App posts to
  Checks is sanitised" below).
- ✅ **A review of a commit that is no longer the PR's head is skipped at
  checkout.** The webhook
  handler passes the delivery's `repository.id` and `pull_request.head.sha`
  on (a delivery without a usable one is acknowledged and ignored). After the
  checkout, `git rev-parse HEAD` must equal that sha: `refs/pull/<n>/head`
  moves with every push, so when it does not, `runReview` logs
  `skipping <owner>/<repo>#<n>: checkout is at <sha>, delivery is for <sha>`,
  runs nothing and posts nothing. A push that outran the delivery has its own
  delivery, which reviews the new head. A review that starts on one commit
  can still finish after a push: the window is the time between the check
  and the end of the review, not zero.
- ✅ **Tolerant indexing.** `indexRepository` mirrors
  `trelix-review.yml`'s own `if ! trelix index .; then ::warning ...; fi`
  pattern — an indexing failure degrades findings to structural-only
  rather than blocking the review.
- ✅ **Redelivery backstop.** GitHub webhooks get no automatic retry past
  their own transient-failure window — a non-2xx response marks the
  delivery permanently failed. `src/scripts/redeliver-webhook-deliveries.ts`
  (via `getAppJwt`'s App-level JWT auth — distinct from every other call
  in this codebase, which uses an installation token) redelivers failed,
  sufficiently-aged deliveries. Runs on both a 6-hour schedule and
  `workflow_dispatch` — see "Deploying on Render" below.
- ✅ **Payload size limit.** The webhook route caps request bodies at 25MB
  — GitHub's own documented webhook payload cap — rejecting oversized
  bodies with `413` during parsing rather than buffering an arbitrarily
  large request into memory. This matters because signature verification
  happens *after* body parsing, so the size limit is the only defense
  against a sender who doesn't know the webhook secret sending a
  deliberately huge payload.
- ✅ **Subprocess timeout.** `runReviewCli` passes a 5-minute `timeout` to
  the `trelix review` shell-out; Node kills the child process (`SIGTERM`)
  and the call rejects if it hangs past that — a slow/stuck review no
  longer ties up server resources indefinitely.
- **Not claimed: GitHub Marketplace listing.** This App is installable
  and hardened, not Marketplace-verified — Marketplace paid-app listing
  has its own separate business/adoption requirements that are out of
  scope for this engineering work.

### Token scopes

A bare installation token reaches every repository of the installation with
every permission the App holds. The App never asks for one: each review mints
three tokens (`src/auth.ts`, `getPurposeTokens`), each limited to the single
repository the pull request is in (`repository_ids: [<repository.id>]`) and to
what one job needs. They are used in different places, so a compromised git or
review child cannot forge Checks or read another repository.

| Token | Permissions | Used by | What it is for |
|---|---|---|---|
| `checkout` | `contents: read`, `metadata: read` | the `git` child only, through the askpass helper (`TRELIX_GIT_TOKEN`) | `git fetch` of `refs/pull/<n>/head` |
| `review` | `pull_requests: read`, `metadata: read` | the `trelix review` child only (`GITHUB_TOKEN`) | `GET /repos/{owner}/{repo}/pulls/{n}/files`, the only GitHub call `trelix review --pr --json` makes |
| `poster` | `checks: write`, `metadata: read` | this process only (Octokit) | `POST /repos/{owner}/{repo}/check-runs`, the only call this process makes |

- `trelix review --post-comments` would also call `GET /pulls/{n}` and
  `POST /pulls/{n}/reviews`, which need `pull_requests: write`. The App never
  passes that flag, so the `review` token has no write permission.
- `@octokit/auth-app` caches a token per installation, repository ids and
  permission set (`optionsToCacheKey`), so the three purposes never share a
  cached token and a repeat review of the same repository reuses them for up to
  59 minutes. `tests/auth.test.ts` asserts the body of every token request,
  and `tests/scoped-tokens.test.ts` that each consumer gets the token minted for
  it.
- No installation token is logged: `tests/scoped-tokens.test.ts` runs a whole
  review (clean, skipped, failing, refused by the API) and looks for the tokens
  and the App JWT in everything printed to the console.
- **The App's registration is changed by its owner.** `manifest.yml` now asks
  for `pull_requests: read` (nothing in the App comments on or edits pull
  requests, and the `pull_request` event needs only read), but a manifest is
  read only when an App is created. For the already registered App, change
  Settings -> Developer settings -> GitHub Apps -> Permissions & events ->
  Pull requests from *Read & write* to *Read-only*. The tokens above are
  requested with `read`, which works before and after that change, and
  GitHub asks installers to approve added permissions, not removed ones. After
  deploying, open a test pull request and check that a `trelix Code Review`
  Check appears: a token request GitHub refuses (for example a permission the
  App does not hold) is logged as `[webhook] review failed` and posts no Check.

### Production deployment notes

- Run behind HTTPS (a reverse proxy or platform-provided TLS termination)
  — GitHub's webhook deliveries and the manifest's `hook_attributes.url`
  require it.
- `GITHUB_APP_PRIVATE_KEY`/`GITHUB_WEBHOOK_SECRET` must come from your
  platform's secret manager, never a committed file — `src/config.ts`
  reads them from env only and throws at startup if either is missing.
- `trelix` (the CLI) and a Python 3.12+ runtime must be present in the
  deployment image/environment — `review-runner.ts` shells out to it by
  name via `PATH`. `Dockerfile` (below) builds exactly this.
- Node 24 or newer (`engines.node` is `>=24`; Node 20 reached end of life on
  2026-04-30). The `Dockerfile` takes the `node` binary from
  `node:24-bookworm-slim`, the same image its build stages use, instead of
  running a NodeSource install script, and the CI job "Docker build" starts
  that binary in the built image and fails unless it reports Node 24. Outside
  the image, run a Node 24 LTS build.
- **Untrusted PR content.** Every PR is checked out from an outside author,
  so the service hardens that checkout:
  - `TRELIX_WALKER_FOLLOW_SYMLINKS=false` is set in the `Dockerfile` (and
    `render.yaml`). trelix follows symlinks out of the repo by default, so
    without it a symlink committed in a PR would make `trelix index` read
    files from the host. If you deploy without this image (or override the
    variable on your platform), set it yourself. `src/child-env.ts` also
    forces it to `false` for both `trelix` children, whatever the host
    passes in.
  - `repo-checkout.ts` checks out with `core.symlinks=false` (committed
    symlinks become plain files holding the link text) and deletes any
    `.trelix` entry from the fresh workspace, so a PR cannot supply its
    own index database.
  - **Every child process gets an allow-listed environment.** `src/child-env.ts`
    builds each one from an empty object and copies in only the names it
    lists, so `GITHUB_APP_PRIVATE_KEY`, `GITHUB_WEBHOOK_SECRET`, the
    platform's `RAILWAY_*` variables and any other name not listed never reach
    a child:
    - `git` (`init`, `remote add`, `fetch`, `checkout` and `rev-parse`): `PATH`, `LANG`,
      an empty `HOME` and `XDG_CONFIG_HOME`, `GIT_ASKPASS`, `TRELIX_GIT_TOKEN`
      and the isolation variables below.
    - `trelix index`: `PATH`, `LANG`, `HOME`, `XDG_CONFIG_HOME` (where trelix
      looks for the operator's `trelix/env` file), every provider variable
      trelix's config reads (for example `AZURE_API_KEY`, `AZURE_ENDPOINT`,
      `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AWS_ACCESS_KEY_ID`), the names
      the provider SDKs read for themselves, and every `TRELIX_*` variable
      except `TRELIX_GIT_TOKEN`. The SDK names include the common credential
      chains of a deployment without static keys, though not every name the
      SDKs read (see "Not passed" below): AWS role and web-identity login
      (`AWS_ROLE_ARN`, `AWS_WEB_IDENTITY_TOKEN_FILE`, `AWS_SESSION_TOKEN`),
      ECS/EKS container credentials (`AWS_CONTAINER_CREDENTIALS_FULL_URI` and
      `_RELATIVE_URI`, `AWS_CONTAINER_AUTHORIZATION_TOKEN` and `_FILE`), a
      shared credentials file (`AWS_SHARED_CREDENTIALS_FILE`) and region
      (`AWS_DEFAULT_REGION`), Vertex (`GOOGLE_APPLICATION_CREDENTIALS`),
      Anthropic identity federation, Azure AD tokens, and gateway base URLs
      such as `OPENAI_BASE_URL`. The full lists are in `child-env.ts`, and a
      Python test keeps them in step with trelix's config and with the names
      the installed provider SDKs are scanned for. Three `GIT_CONFIG_*`
      variables are also set, see the git bullet below. A pointer such as
      `AWS_WEB_IDENTITY_TOKEN_FILE` hands the child the identity itself, so
      scope the role behind it to model inference only.
    - `trelix review`: the same, plus the `review` installation token (see
      "Token scopes") as `GITHUB_TOKEN`. Only this child receives it.

    Names are matched exactly and case-sensitively. **Not passed**, because
    they are on no list: proxy variables (`HTTP_PROXY`, `HTTPS_PROXY`,
    `NO_PROXY` and their lowercase spellings), CA-bundle variables
    (`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`,
    `NODE_EXTRA_CA_CERTS`, `AWS_CA_BUNDLE`), `AWS_CONFIG_FILE`, and every
    other platform variable. Some provider-SDK names are withheld on purpose:
    webhook-signing and admin keys (`ANTHROPIC_WEBHOOK_SIGNING_KEY`,
    `OPENAI_WEBHOOK_SECRET`, `OPENAI_ADMIN_KEY`), Azure service-principal
    secrets (`AZURE_CLIENT_SECRET` and the certificate and federated-token
    names), and the names of services trelix has no client for (S3, Azure
    services other than Azure OpenAI, Google search).

    **The lists do not cover everything the SDKs read.** These are also not
    forwarded: `AWS_BEARER_TOKEN_BEDROCK` (a secret: Bedrock API-key
    authentication), `AWS_EC2_METADATA_DISABLED`, `OPENAI_ORG_ID`,
    `OPENAI_PROJECT_ID`, `OPENAI_API_TYPE`, `OPENAI_CUSTOM_HEADERS` and
    `GOOGLE_GENAI_USE_VERTEXAI`. A deployment that relies on one of them (a
    Bedrock API-key user, for one) is not covered: typically authentication
    fails, `trelix review` exits 3 and the PR gets a neutral check instead of
    a review. Open an issue, or add the name to the allow-list in
    `child-env.ts`; `tests/unit/test_github_app_child_env_contract.py` pins
    that list to the names the installed SDKs are scanned for, so a name
    outside that scan also needs a reviewed edit to that test.

    Every `TRELIX_*` variable passes to the `trelix` children, so do not keep
    an App secret under a `TRELIX_` name. Env filtering is leak hardening,
    not a sandbox: a child running as the same user may be able to read the
    App process's environment through the operating system (for example
    `/proc/<pid>/environ` on Linux).
  - **git is isolated from the host and from the PR.** Each git child runs
    with `GIT_CONFIG_GLOBAL=/dev/null`, `GIT_CONFIG_SYSTEM=/dev/null`,
    `GIT_CONFIG_NOSYSTEM=1`, `GIT_TERMINAL_PROMPT=0`,
    `GIT_ALLOW_PROTOCOL=https` and `HOME`/`XDG_CONFIG_HOME` set to an empty
    directory. Every git command also gets
    `-c safe.bareRepository=explicit -c protocol.file.allow=never -c credential.helper=`,
    and the repository is created with `git init --template=` (an empty
    template). `safe.bareRepository=explicit` matters because a PR can commit
    a bare repository into its tree as ordinary files: a git command run from
    inside it would otherwise adopt that repository and could run the command
    in its `core.fsmonitor`. The allowed protocol can be widened only by the
    `allowProtocol` option of `checkoutPullRequest` (used by the tests to
    fetch from a local repository), never from the environment.
    `-c safe.bareRepository=explicit` needs git 2.38 or newer; an older git
    silently ignores it. The image installs Debian's git, which is newer.
    `-c protocol.file.allow=never` is redundant with
    `GIT_ALLOW_PROTOCOL=https` (the environment variable overrides it, which
    is also why the tests' `allowProtocol: "file"` works); it is kept as a
    second guard.
  - **The git calls `trelix` makes itself get almost none of that.** This
    isolation covers the git commands the service runs (`init`, `remote add`,
    `fetch`, `checkout`, `rev-parse`). The ones inside `trelix` (`git_linker.py`,
    `diff_parser.py` and `provenance.py`, each run with `cwd=repo_path`) get
    no `-c` config, no empty `HOME` and no `GIT_CONFIG_GLOBAL=/dev/null`, and
    rely on the checkout root being their working directory. The one defence
    they do get is `safe.bareRepository=explicit`, passed to the `trelix`
    children as `GIT_CONFIG_COUNT=1`, `GIT_CONFIG_KEY_0` and
    `GIT_CONFIG_VALUE_0` (command-scope config, so it outranks any config
    file; same git 2.38 floor), which keeps them from adopting a bare
    repository found implicitly.
  - The `GIT_ASKPASS` helper and git's `HOME` live in a separate
    `trelix-review-aux-*` temp directory outside the checkout. `cleanup()`
    removes it with the checkout and `sweepStaleWorkspaces` removes leftovers
    of both at boot. A PR that tracks a file named `.git-askpass.sh` therefore
    checks out normally.
- **Everything the App posts to Checks is sanitised.** The text of a Check
  comes from an LLM that reads attacker-written pull requests, so it may be
  prompt-injected, and a Check is shown to maintainers. `src/sanitize.ts` makes
  sure it cannot show them an image (a tracking pixel), a link or bare URL, an
  @mention, raw HTML or hidden text. `createCheckRun` in `review-runner.ts` is
  the only place that creates a Check run, and it passes the whole `output`
  through `sanitizeCheckOutput` first, so `postCheckRun`,
  `postIncompleteCheckRun`, `postReviewFailureCheckRun` and any poster added
  later are covered. A test fails if another `checks.create` call appears.
  - **What is covered.** The output `title` and `summary`, and each
    annotation's `path`, `title` and `message`. Only those fields are copied
    into the request. Line numbers are not text and are not checked.
  - **The rules**, in this order, each a plain rewrite of the text with no
    Markdown parser (so crafted input cannot desynchronise it from GitHub's):
    1. Control characters, zero-width characters, bidi overrides and
       isolates, the Unicode tag block (U+E0000 to U+E007F, where prompt
       smuggling hides text), variation selectors, invisible fillers and
       unpaired surrogates are removed. CR, CRLF, U+2028 and U+2029 become LF.
    2. HTML comments are removed, including the HTML5 forms `<!-->`, `<!--->`
       and `--!>`. An unterminated `<!--` removes the rest of the text, which
       is what a renderer would hide.
    3. Markdown images (`![alt](url)`) are dropped.
    4. A run of more than four combining marks is cut to four (zalgo text
       bleeds over the lines around it).
    5. URLs: every `://` becomes `[:]//` (`https[:]//host/path`, backslash
       escapes included) and `www.` becomes `www[.]`. The text stays readable
       and is no longer a link.
    6. Link syntax is broken: `](` becomes `] (`, so `[text](url)` reads
       `[text] (url)`, and `]:` becomes `] :`, so a reference definition, which
       renders as nothing, is shown.
    7. Every `@` becomes a fullwidth `@` (`＠octocat`, `me＠host.example`),
       which covers user mentions, team mentions and email addresses. It is
       every `@`, not only one before a letter or digit: GitHub also links
       an address whose domain starts with `-`, `_` or `.`, or with a
       backslash escape of one of them (`a@-b.example`, `a@\.b.example`), so
       no rule that guesses which `@` is harmless is safe.
    8. `<` and `>` become the fullwidth `＜` and `＞`, so `List<String>`
       reads `List＜String＞` in prose, in backticks and in a plain-text
       annotation, and no tag can form. (An entity such as `&lt;` would show
       literally inside a code span.) A character reference such as `&#64;`
       or `&copy;` is shown as text (`&amp;#64;`), because the renderer would
       decode it. A lone `&`, and `&amp;`, `&lt;` and `&gt;` as written, are
       left alone.
    9. Outer whitespace is trimmed and the text is cut to its limit with a
       trailing `…`, never inside a character reference or a surrogate pair.

    The output is a fixed point: sanitising sanitised text changes nothing,
    which is why the poster and `toAnnotations` can both apply it.
  - **Limits.** Title 140 characters, summary 4,000, annotation message 2,000,
    path 1,024, and 50 annotations per check (GitHub's limit per request). A
    message that is empty once sanitised is posted as `(no details provided)`,
    because an annotation needs a message and one bad annotation would fail
    the whole request.
  - **Annotation paths.** An annotation is dropped when its path is not a
    string, is empty or over the limit, has a line break, is absolute (`/`,
    `\`, `C:\`), has a `..` segment, or looks like markup or a URL (`<`, `>`,
    `://`, `](`). Hidden characters are removed from a path first, so a
    zero-width character inside `..` does not hide it. Other characters,
    including `@` and `[id]`, stay: GitHub matches the path against the
    repository's files. The Check's verdict and issue count come from the
    findings, not from the annotations, so a dropped or over-limit annotation
    cannot turn a failure into a success; the summary says how many findings
    went without an annotation.
  - **What it costs in readability.** The rules are context free, so code is
    treated like prose, even inside backticks: `@Override` reads `＠Override`
    (so does a shell `"$@"` or an import alias `@/lib`), `Optional<String>`
    reads `Optional＜String＞` (fullwidth signs are wider than ASCII and do
    not paste back into code), `handlers[i](e)` reads `handlers[i] (e)` and
    `x[1]: int` reads `x[1] : int`. Removing hidden characters also removes
    the invisible characters that real text uses: an emoji loses its variation
    selector (a red heart becomes a plain one), a joined emoji sequence (a
    family) falls apart into its emoji, and the zero-width joiner and
    non-joiner that Persian, Indic and other scripts use for shaping are
    dropped.
  - **Not covered.** Issue, pull request and commit references (`#123`,
    `owner/repo#1`, a commit SHA) and emoji shortcodes still render as GitHub
    renders them. Diagram and math blocks (a `mermaid` fence, `$...$`) get the
    same rewrites as any text, so their URLs are defanged, but a scheme-relative
    `//host` (which has no `://`) is left as it is. The escapes assume GitHub
    renders the annotation message as Markdown, which is not verified. If it
    shows plain text, nothing changes for `<`, `>` and `@` (they are replaced,
    not escaped as entities), and only a character reference typed in a
    finding (`&#64;`) shows as `&amp;#64;`. Line numbers are passed
    through as they are. The Actions
    workflow (`.github/workflows/trelix-review.yml`) does not use this
    sanitiser.
- Logs (`console.error`/`console.warn` on review/indexing failures)
  currently go to stdout/stderr only; wire your platform's log
  aggregation on top rather than expecting structured logging from this
  service directly.

### Files

- `manifest.yml` — GitHub App manifest for the [manifest registration
  flow](https://docs.github.com/en/apps/sharing-github-apps/registering-a-github-app-from-a-manifest).
  Declares the App's permissions (`pull_requests: read`, `checks: write`,
  `contents: read`; the App never comments on pull requests, so it has no
  `pull_requests: write`) and subscribes to the `pull_request` event. See
  "Token scopes" for what the already registered App's owner must change.
- `src/server.ts` — Express entry point (`/health`, `/webhooks/github`).
- `src/webhook.ts` — verifies `X-Hub-Signature-256`, then routes
  `pull_request` `opened`/`synchronize`/`reopened` deliveries (mirrors the
  Actions workflow's trigger), invokes the review runner with the delivery's
  repository id and head sha (`src/commit-id.ts` checks the sha).
- `src/child-env.ts` — the allow-listed environment of each child process
  (`git`, `trelix index`, `trelix review`); see "Untrusted PR content"
  above.
- `src/review-runner.ts` — mints the three purpose-scoped installation
  tokens, clones the PR's actual head into a fresh workspace
  (`repo-checkout.ts`), skips the review if that is no longer the delivery's
  head commit, indexes and reviews it via the `trelix` CLI, and posts the
  findings as a GitHub
  Check run (`toAnnotations`/`postCheckRun` — a TypeScript port of the
  same mapping logic in `trelix-review.yml`'s `github-script` step). A
  review that exits 4 (part of the diff unreviewed) is posted by
  `postIncompleteCheckRun`.
- `src/review-outcome.ts` — the exit codes 3 and 4 (kept equal to
  `src/trelix/cli/main.py` by `tests/unit/test_review_exit_code_contract.py`),
  the conclusion rule (`reviewConclusion`), the strict reader of the
  outcome record (`parseOutcomeRecord`, `readOutcomeRecord`), the private
  directory the record is written to (`createOutcomeLocation`) and the summary
  text (`buildIncompleteSummary`). The workflow carries its own copy of the
  rule, the reader and the summary; both are tested against the one table in
  `tests/fixtures/review-conclusion-cases.json`.
- `src/sanitize.ts` — the sanitiser every string posted to Checks goes
  through (`sanitizeCheckOutput`, applied in `review-runner.ts`'s
  `createCheckRun`); see "Everything the App posts to Checks is sanitised".
- `src/repo-checkout.ts` — clones a single PR's head into a per-request
  temp workspace via an installation token; see "Clone-on-demand" above
  for the security properties this enforces.
- `src/auth.ts` — installation-token minting, one token per purpose
  (`getInstallationToken`, `getPurposeTokens`; see "Token scopes") and
  App-level JWT minting (`getAppJwt`, used only by the redelivery script)
  via `@octokit/auth-app`, one cached `AuthInterface` per `AppConfig`.
- `src/commit-id.ts` — `isCommitId`, the check that a webhook's head sha and
  what `git rev-parse HEAD` prints are full lowercase hex commit ids.
- `src/scripts/redeliver-webhook-deliveries.ts` — the redelivery backstop;
  run via `npm run redeliver-failed-webhooks` or
  `.github/workflows/redeliver-failed-webhooks.yml`.
- `src/config.ts` — reads `GITHUB_APP_ID`/`GITHUB_APP_PRIVATE_KEY`/
  `GITHUB_WEBHOOK_SECRET` from env only, per this repo's "never hardcode
  secrets" convention.
- `Dockerfile` — builds this service + the `trelix` CLI it shells out to
  into one image; see "Deploying on Render" below.
- `render.yaml` — Render Blueprint for the free-tier deploy described
  below.

### Registering the App

1. Edit `manifest.yml`: replace the placeholder `https://trelix.example.com`
   URLs with your deployed service's real HTTPS origin.
2. Register via the [manifest flow](https://docs.github.com/en/apps/sharing-github-apps/registering-a-github-app-from-a-manifest)
   (create a temporary HTML form that POSTs the manifest JSON to
   `https://github.com/settings/apps/new`, or your organization's
   equivalent settings page).
3. GitHub redirects back with a one-time `code` (no `redirect_url`/
   `setup_url` is configured, so this is a plain query-string param on
   GitHub's own confirmation page, not a redirect into this service).
   Exchange it directly for the App's credentials:
   ```bash
   curl -X POST "https://api.github.com/app-manifests/${CODE}/conversions"
   ```
   The response includes the App ID, a generated private key, and (once
   you set a webhook secret in the App's settings) is where you'll also
   find the webhook secret.
4. Set `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY`, `GITHUB_WEBHOOK_SECRET` as
   environment variables (never commit them — see `.gitignore`'s `.env`
   entry).

### Deploying on Railway

Migrated off Render: Render fronts every service through Cloudflare
non-configurably, and Cloudflare's WAF blocked real GitHub `pull_request`
webhooks whose body contained code-like content (curl examples, code
fences) — a documented, unresolved false-positive class, not fixable from
our side. Railway has no default content-inspecting WAF, so this class of
failure can't recur there. `render.yaml`/the section this replaced is kept
only as a historical fallback reference until Render is fully decommissioned.

`Dockerfile` (same one, unchanged) builds this service and the `trelix`
CLI it shells out to into one image. Configure via the Railway dashboard
or CLI — **do not** add a `railway.json`/`railway.toml`; that config-as-code
format is deprecated with a 2026-12-01 cutoff in favor of a new
TypeScript/Python/Go IaC system, and our config needs are modest enough to
skip it entirely:

- **Root Directory:** repo root (blank/`.`).
- **Dockerfile path:** `infra/github-app/Dockerfile`, set via the
  `RAILWAY_DOCKERFILE_PATH` service variable (or the dashboard's
  Dockerfile-path field) — this also switches the effective builder to
  `DOCKERFILE` automatically; there's no separate "Docker" builder enum
  value to pick.
- **Branch:** `main` — despite this repo's convention of deploying `develop`
  everywhere else, Railway's auto-deploy trigger for this service is
  actually configured against `main` (confirmed live via `railway status
  --json`'s `meta.branch` field, 2026-09-25 — this doc previously said
  `develop`, which was stale).
- **If deploys stop firing on real merges:** confirmed live on 2026-09-25 —
  the GitHub↔Railway connection can silently disconnect (deploy history
  showed a multi-day gap with zero attempts, not failures, despite many
  qualifying pushes). `railway service source connect --repo
  sairam0424/trelix --branch main --service trelix-github-app` recreates
  the deployment trigger and immediately kicks off a fresh build — but this
  is a one-shot fix, not a persistent one: it only reuses the plain "login
  with GitHub" OAuth connection, which is enough to trigger a single deploy
  but not enough to sustain auto-deploy-on-push. That needs Railway's own
  GitHub App to actually be *installed* on the account (not just the OAuth
  login) — check `github.com/settings/installations` for a **Railway**
  entry; if it's missing entirely (found live on 2026-09-26 — it wasn't
  there at all, despite the OAuth login working fine), install it from
  `github.com/apps/railway-app`, grant it access to this repo, then in
  Railway's Source settings reconnect the branch — "Branch connected to
  production" and "Auto deploys when pushed to GitHub" only appear once the
  App install actually exists.
- **Health check:** path `/health`, generous timeout (300s) to cover a
  cold start plus a real `git clone` + `trelix index`/`review` burst.
- **Sleep (Serverless):** the Free plan *requires* `sleepApplication: true`
  for any service without a cron schedule — it cannot be disabled short of
  upgrading off Free. This means the same cold-start-drops-a-delivery risk
  Render's 15-minute spin-down had is unavoidable here too, so the same two
  mitigations apply, just repointed at the Railway URL:
  - **Keep-warm pinger — [cron-job.org](https://cron-job.org):** a free
    job hitting `https://<your-service>.up.railway.app/health` every
    10–14 minutes.
  - **Redelivery backstop:** `.github/workflows/redeliver-failed-webhooks.yml`
    (schedule + `workflow_dispatch`) — unchanged, hosting-agnostic. Needs
    `TRELIX_APP_ID`/`TRELIX_APP_PRIVATE_KEY` as repo secrets (Settings →
    Secrets and variables → Actions) — renamed from `GITHUB_APP_ID`/
    `GITHUB_APP_PRIVATE_KEY` in PR #337; the env var *names* `config.ts`
    itself reads are unchanged, only the GitHub Actions secret-store names
    changed.
- **Replica/Usage Limits:** set a generous (not aggressive) Replica Limit
  so a real review burst doesn't crash the instance, and only a *soft*
  (notify-only) Usage Limit — no hard billing cutoff that could kill the
  service unexpectedly. Configure via the dashboard; the underlying
  `usageLimitSet` mutation is billing-workspace-scoped, not project-scoped,
  so it's not something to script per-project.
- **Free-tier deploy freeze:** Railway blocks Free-plan deploys roughly
  8 AM–8 PM `America/Los_Angeles` daily, regardless of the service's own
  region. A push that would trigger an auto-deploy during that window
  simply fails until the window closes — worth knowing before assuming a
  merged PR redeployed.

Env vars (`GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY`, `GITHUB_WEBHOOK_SECRET`)
are the same shape as before. LLM synthesis uses `TRELIX_LLM_PROVIDER=azure`
plus `AZURE_API_KEY`/`AZURE_ENDPOINT` (reusing the main app's existing,
already-working Azure credentials) instead of a placeholder
`OPENAI_API_KEY` — no new credential was provisioned for this.

### Local development

Needs Node 24 or newer (`engines.node`).

```bash
npm install
npm run typecheck
npm run build
npm test

GITHUB_APP_ID=... GITHUB_APP_PRIVATE_KEY=... GITHUB_WEBHOOK_SECRET=... npm run dev
```

`trelix` (the CLI) must be installed and on `PATH` wherever this service
runs — `review-runner.ts` shells out to it directly.
