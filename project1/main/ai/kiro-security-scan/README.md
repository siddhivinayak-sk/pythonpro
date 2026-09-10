# kiro-security-scan

Security analysis of **AI artifacts and documents** using the **Kiro headless
CLI** as the LLM reviewer. It walks a directory, filters files, sends each
eligible file to Kiro for a security review against the **OWASP Top 10 for LLM
Applications (2025)** and the **OWASP Top 10 for Agentic Applications
(ASI01–ASI10)**, collects strict-YAML findings, applies a false-positive
whitelist, and produces an HTML report plus a machine-readable YAML file.

All scanner logic lives in a single file: [`main.py`](./main.py).

## What it detects

- **OWASP LLM Top 10 (2025):** prompt injection, sensitive information
  disclosure, supply chain, data/model poisoning, improper output handling,
  excessive agency, system prompt leakage, vector/embedding weaknesses,
  misinformation, unbounded consumption.
- **OWASP Agentic Top 10 (ASI01–ASI10):** goal hijack, tool misuse, identity &
  privilege abuse, agentic supply chain, unexpected code execution, memory &
  context poisoning, insecure inter-agent comms, cascading failures,
  human–agent trust exploitation, rogue agents.
- **AI-artifact specifics:** Skills, Powers, steering/instruction files,
  `mcp.json` (e.g. `--trust-all-tools`, secrets in env, unpinned servers),
  prompts/templates, hooks, agent configs.
- **Classic issues with CWE ids:** hardcoded secrets (CWE-798), XSS (CWE-79),
  command injection (CWE-78), code injection (CWE-94), path traversal (CWE-22),
  SQL injection (CWE-89), etc.

## Requirements

- Python 3.9+ and `pip install -r requirements.txt` (just PyYAML).
- Kiro CLI installed (`kiro-cli`) — `curl -fsSL https://cli.kiro.dev/install | bash`.
- A Kiro **Pro / Pro+ / Pro Max / Power** API key exported as `KIRO_API_KEY`
  (headless mode requires an API key).

## Quick start (local)

```bash
pip install -r requirements.txt
export KIRO_API_KEY="your-key"            # PowerShell: $env:KIRO_API_KEY="your-key"

# Scan a directory with defaults
python main.py --input ./sample-artifacts --report ./out/report.html

# Or drive everything from a config file
python main.py --config ./config.yaml
```

Exit codes: **0 = PASS**, **1 = FAIL** (any open CRITICAL/HIGH finding),
**2 = setup error** (e.g. missing `KIRO_API_KEY` or `kiro-cli`).

## Use as a library

The single entry method returns a boolean so an application can gate on it:

```python
from main import run_scan

passed = run_scan(config_path="config.yaml")   # True = PASS, False = FAIL
# rich detail is in the HTML report + YAML file produced as side effects
```

You can also pass overrides directly:

```python
passed = run_scan(input_dir="./docs", model="auto",
                  report_path="out/report.html")
```

## Docker (run with just KIRO_API_KEY)

```bash
docker build -t kiro-security-scan .

docker run --rm \
  -e KIRO_API_KEY="$KIRO_API_KEY" \
  -v "$(pwd)/sample-artifacts:/scan/input:ro" \
  -v "$(pwd)/out:/scan/out" \
  kiro-security-scan
```

PowerShell:

```powershell
docker run --rm `
  -e KIRO_API_KEY="$env:KIRO_API_KEY" `
  -v "${PWD}\sample-artifacts:/scan/input:ro" `
  -v "${PWD}\out:/scan/out" `
  kiro-security-scan
```

The container exits 0/1/2 as above, so it plugs straight into CI.

## Configuration

Full options are documented inline in [`config.yaml`](./config.yaml). Highlights:

| Key | Meaning |
| --- | --- |
| `input_dir` | Directory to scan. |
| `exclude` | List of names / dirs / glob patterns to skip (e.g. `node_modules`, `**/*.png`). |
| `include_extensions` / `include_filenames` | Which file types/names are eligible for review (defaults cover md/html/json/yaml/source/prompt files, `mcp.json`, etc.). |
| `model` | LLM model id, or `auto` (default). List ids with `kiro-cli chat --list-models`. |
| `prompt_override` | Inline instructions string **or** a path to an instructions file, replacing the built-in analysis instruction block (used by both agent and inline mode). |
| `whitelist` / `whitelist_path` | False-positive entries (see below). |
| `report_path` | HTML report output path. |
| `yaml_output_path` | Machine-readable YAML findings path (set null to skip). |
| `fail_severities` | Severities that fail the scan when open (default CRITICAL, HIGH). |
| `use_agent` / `agent_name` | Agent mode (default on): write instructions once to a Kiro custom agent and send only a small per-file prompt via `--agent`. |
| `content_mode` | `reference` (default): Kiro reads each file itself (no embedding/splitting); `embed`: content is sent in the prompt (large files split). |
| `timeout_seconds`, `max_retries`, `retry_backoff_seconds` | Transient-error handling for the CLI. |

### Agent mode (how the prompt is sent)

The large static analysis instructions (OWASP taxonomies, severity rubric,
output contract) are the same for every file. Rather than resend them with each
call, the scanner writes them once to a Kiro **custom agent** definition
(`.kiro/agents/<agent_name>.json` referencing a `file://` instructions file) in
a temporary working directory, then invokes:

```
kiro-cli chat --no-interactive --agent <agent_name> [--model <id>] --trust-tools=read,grep "<per-file prompt>"
```

The taxonomy/rules/output-contract come from the agent instructions. The model
is selected with the CLI's `--model` flag when not `auto`.

### How file content reaches the model

**Reference mode (default, `content_mode: reference`):** Kiro reads each target
file itself. Before each call the agent's `resources` is pointed at the file
(`file://<absolute path>`), which eager-loads its full content into context at
startup; the agent also has the `read` tool trusted with an `fs_read` permission
scoped to the input directory, so it can read the whole file (including ranged
reads for large files). Nothing is embedded in the prompt and large files are
**not** split. This is the most robust option and has no per-file size limit
beyond the model's context window.

**Embed mode (`content_mode: embed`, and automatic fallback):** the file content
is embedded in the prompt. To stay under the OS command-line argument length
limit (`Argument list too long` / E2BIG, ~128 KB per argument on Linux), files
whose content exceeds the safe budget are split into line-aligned segments, each
analyzed separately. Reported line numbers are always the file's real line
numbers, and findings duplicated across a segment boundary are de-duplicated.

If reference mode fails for any reason (e.g. the CLI build doesn't support
custom agents), the scanner automatically falls back to embed mode. The agent is
locked down with deny-all permissions except read access scoped to the input
tree, so the reviewer can't act on instructions embedded in scanned files.

If a CLI build rejects `--agent`, the scanner automatically falls back to
inline mode (instructions prepended to each prompt). You can force inline mode
with `use_agent: false`.

Note: headless `kiro-cli` calls are stateless and Kiro documents no prefix
caching, so agent mode does not reduce per-call token cost — it keeps
invocations small and the instructions centralized and version-controlled.

CLI flags (`--input`, `--exclude`, `--model`, `--prompt`, `--whitelist`,
`--report`, `--yaml-out`, `--kiro-cli`, `--timeout`, `--max-retries`, `-v`)
override config-file values.

## Output

**Strict YAML** (`yaml_output_path`) — one record per finding, programmatically
parseable:

```yaml
findings:
  - file: docs/example.md
    line: 12
    column: 1
    vulnerability_id: LLM02
    vulnerability: Hardcoded API key
    severity: CRITICAL
    description: A live-looking API key is embedded in the document.
    mitigation: Remove the secret; load from a secret manager or env var.
    whitelisted: false
    whitelist_reason: ""
    fingerprint: a1b2c3d4e5f60718
```

**HTML report** (`report_path`) — includes the scan date/time, metadata,
per-severity summary of **open** and **whitelisted** counts, a PASS/FAIL
verdict, and full tables (file, line:column, standard vuln id, vulnerability,
severity, description, mitigation) for both open and whitelisted findings. It
also shows the **total time taken**, the number of **LLM requests** made, and
**credits consumed**.

### Time and credits

- **Total time taken** and **LLM requests** (number of `kiro-cli` calls) are
  always measured and reported, in the log line at the end and in the HTML
  report (near the verdict and in the metadata table) and YAML.
- **Credits consumed:** the Kiro CLI does not expose credit usage in headless
  mode, so credits show as *unavailable* by default (track usage in the billing
  dashboard at app.kiro.dev). If you set `credits_command` (config or
  `--credits-command`) to a command that prints the current numeric balance,
  the scanner runs it before and after the scan and reports the difference.

## Whitelisting false positives

Add entries to [`whitelist.yaml`](./whitelist.yaml). Match by exact
`fingerprint` (copied from the YAML output) or by any subset of `file`,
`vulnerability_id`, `vulnerability`, `severity`. Whitelisted findings are still
shown in the report but do **not** affect the PASS/FAIL verdict.

## Reproducibility

The built-in prompt fixes the taxonomy, severity rubric, output schema, and
sort order, and instructs the model to report only evidence-backed findings.
LLM output is not guaranteed bit-identical across runs, but this design keeps
results stable for the same content. Use `model: auto` (default) or pin a
specific model id for tighter consistency.

## Notes

- Only text-like files under the size cap are reviewed; binaries are skipped
  because an LLM can't meaningfully analyze them.
- File content is treated strictly as data — the prompt instructs the reviewer
  never to follow instructions embedded in scanned files (anti prompt-injection).
- The reviewer session runs least-privilege (`--trust-tools=read,grep`).
