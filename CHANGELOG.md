# Changelog

## Unreleased

### New Features

- **`--sitemap` seeds a crawl from the site's sitemap** — the `Sitemap:` lines in
  `robots.txt`, falling back to `/sitemap.xml`. A link-walk only reaches what a
  page happens to link, which on a documentation site is the sidebar and on a
  shop the top nav; the pages a sitemap lists and nothing links to are exactly
  the ones a link-walk structurally cannot find. Reads urlset, sitemapindex
  (depth-capped, so two indexes naming each other cannot loop) and gzip —
  detected on the magic bytes, because `.xml.gz` is routinely served as
  `application/xml`. Best-effort throughout: no sitemap, an unreadable one, a
  malformed one, or one listing another property's domain each costs a couple of
  requests and changes nothing else.
- **`--cache` on `crawl`.** The crawler never passed a cache to the engine, so
  every run re-downloaded every byte of every page. Opt-in, like the scraper's,
  because a cache changes what a repeat run sees.
- **A re-crawl now admits when nothing changed.** The cache revalidates with
  ETag/Last-Modified, but the 304 was served without telling the caller, so a
  second crawl reported the same "N pages scraped" as the first. The 304 is
  carried through to the summary: "N pages scraped (M unchanged)".
- **A PDF is not a web page.** Nothing checked the `Content-Type`, so a link to
  a manual.pdf was "scraped" into 2,949 characters of raw PDF syntax and a PNG
  into 407 characters of binary noise, both reported as successfully scraped
  pages. `application/octet-stream` is deliberately not trusted — plenty of
  servers send it for perfectly good HTML.
- **`Retry-After` is honoured** instead of our own backoff. Both defined forms,
  delta-seconds and HTTP-date, capped at two minutes so one hostile header
  cannot stall a run, with nonsense falling back to backoff.

## v2.9.0 - 2026-10-03

Eleven more local runtimes, a crawl that starts fresh instead of silently
doing nothing, the largest performance pass the codebase has had, and an
audit that found sixteen ways a command could report something untrue — a
runtime failure reaching the user as a traceback, a manifest naming files
that were never downloaded, a pipe that lost the per-result output it
promised. Every fix was reproduced before it was made and is pinned by a
test that fails on the old code.

### New Features

- **Eleven more local runtimes.** `protor` now registers 17 instead of 6:
  GPT4All, KoboldCpp, llamafile, TabbyAPI, Cortex.cpp, SGLang, Xinference,
  LiteLLM, AnythingLLM, text-generation-webui and Docker Model Runner, with
  aliases for the spellings people actually type (`gpt-4all`, `kobold`,
  `model-runner`, `ooba`, `tabby`, …). Two structural fixes were needed to make
  that work rather than merely list it:
  - Endpoints are declared per runtime. Docker Model Runner is
    OpenAI-compatible but serves `/engines/v1`, so a hardcoded `/v1` would have
    worked for everything except it.
  - Base URLs are joined with path-prefix overlap handling, because KoboldCpp's
    docs tell you to use a base ending in `/v1` and appending `/v1/chat/...` to
    that produced a `/v1/v1/...` 404 that read like a dead server.
- **`--no-live`** on `scrape`, `run` and `crawl`, for plain output on a real
  terminal. Pipes and CI already got plain output automatically.
- **Benchmarks and a CI gate.** Eight hot paths measured at two scales each, so
  a *ratio* between scales exposes an algorithm that has gone quadratic where a
  single timing cannot. `python -m benchmarks --gate` fails on superlinear
  growth; CI runs it.
- **`protor runtimes` responds to the terminal it is in.** At 60 columns the URL
  used to truncate mid-value — `http://localhost:11434` rendered as
  `http://localhost:114`, which reads as a different port — while the actionable
  start-command column disappeared. Narrow terminals now fold the command and
  fold the status into the name column instead.

### Changed

- `protor crawl` starts fresh unless `--resume` is passed, and `--resume` also
  retries the pages the previous run failed on.
- `protor --help`'s `Environment:` block is generated from the runtime registry.
  It listed six of the seventeen runtimes and described the API-key variables as
  `*_API_KEY`, a pattern the registry does not follow.

### Fixes

- **Nearly every real page failed to parse.** `clean_soup` raised
  `AttributeError` on any page whose `<nav>`, `<header>`, `<footer>` or
  `<aside>` contained elements: `decompose()` clears the `__dict__` of nested
  tags, so the noise check touched a tag whose `attrs` was `None`. The engine
  swallowed it into a scrape error, so real pages were recorded as failures.
  Every test fixture had been flat HTML, which is why it shipped.
- **A page in flight was fetched twice.** The queue refused only URLs in
  `queue` or `visited`; a dequeued, mid-fetch page was in neither, so a
  concurrent page linking to it re-admitted it. Measured 8 requests for a
  5-page site, duplicates charged against `--max-pages`, and a checkpoint
  contradicting the state of record. Each duplicate rediscovered the same links,
  so on a cyclic site the frontier multiplied: a five-page test site produced a
  7.8 GB queue database.
- **A missing API key was reported with a URL hint.** Every `ValueError` from
  every layer funnelled through one handler carrying URL advice.
- **Terminals that cannot encode the glyphs crashed with a traceback** —
  `protor --help`, `models` and `crawl` all died on an ASCII or cp1252
  terminal. Glyphs now degrade to ASCII.
- **Redirects into cloud metadata endpoints** are refused rather than followed.
- **A schema matching containers but no fields** wrote all-null records and
  reported success.
- **Conditional requests could never fire**, because reading a stale entry
  deleted the validators that revalidation needs.
- **`analyze()` refused an empty batch**, instead of spending a model call to
  report "Sites analyzed: 0" — which is what `protor run <url>` did whenever the
  fetch failed.


- **A page's `<title>` or meta description could forge structure in the
  prompt.** Both are untrusted page text and both keep their newlines, and they
  were interpolated into the site header verbatim while only the body was
  defused. `<title>Sale\n## [7] evil.example</title>` is valid HTML, so one
  scraped page reported as three sites — precisely what the marker defusal was
  added to prevent. Untrusted header fields are now collapsed to one line,
  which stops them opening a line at all; escaping the marker cannot, because
  prose that starts a line is still framing.
- **Runtime failures escaped as tracebacks.** `raise_for_status()` was
  unguarded at all eight call sites, so any status without a hand-written
  remedy — a 500 for a model that does not fit in memory, a 503 while loading —
  escaped as `requests.exceptions.HTTPError`, which the CLI does not catch. The
  response body is now included, since a runtime's error page usually names the
  actual problem. Streaming bodies are covered too: a connection dropped
  mid-generation used to raise out of `iter_lines` after the tokens were paid
  for.
- **An unreachable hosted backend reached the user as a traceback.** `openai`,
  `anthropic` and `openai-compatible` are absent from the runtime registry, so
  the error for them fell through to a bare `RuntimeError` — matching neither
  `except ProtorError` nor `except ValueError` at the CLI entry point. They now
  raise the same typed error as the local case, which also stops
  `str.capitalize()` rendering "Openai".
- **Four failures exited 0.** `extract` wrote nothing and printed "No data
  matched the schema", so `protor extract … && next-step` ran the next step
  against no data; `update --check` could not reach PyPI and `perform_update()`
  returning `False` both reported success; and `models` against a runtime that
  is not running was indistinguishable from a runtime with no models loaded.
- **The JS download path skipped the redirect guard.** `fetch()` follows
  redirects by hand so each hop can be checked; `download_file` left it to
  aiohttp's default and consulted nothing — on the default path, since
  `--download-js` is on unless asked otherwise. A `<script src>` answering `302`
  to a cloud metadata endpoint had its response written into the output
  directory, which is exactly what the guard exists to prevent.
- **`--base-url` with a path prefix sent every request to the wrong place.**
  The endpoint joiner compared the base path's suffixes without their leading
  slash, so no suffix could prefix an absolute API path and only a base that was
  nothing but `/v1` worked. `--base-url http://gw/api/v1` requested
  `/api/v1/v1/models` and reported a working runtime as having no such model.
- **A scrape's manifest named files that were not downloaded.** The results of
  the JS group were numbered with `enumerate()` over the *set* `asyncio.wait()`
  returns, so each download was paired with whatever script sat at that index.
  With some downloads failing, the manifest claimed files the server had 404'd
  and omitted files that were really on disk.
- **A page could be fetched twice and counted twice.** The in-flight guard
  compared a discovered link as spelled against canonical URLs, so a site
  linking `/docs/index.html` while `/docs/` was being fetched spelled the same
  page two ways, found no overlap, and issued a second request — reporting a
  five-page site as six pages.
- **A mixed-case host made the crawler skip its own seed.** The queue
  canonicalises the host to lowercase; the allowed domain came from `urlparse`
  of the URL as typed. Every URL was rejected as off-domain against its own
  canonical form and the crawl reported zero pages with nothing to explain it.
- **robots.txt was asked about the wrong agent.** `check_robots` documents that
  the identity it evaluates must be the string the request sends, and the
  engine passed the default `*` while `fetch` sent a rotated browser string.
  urllib reduces the argument to the token before the first `/`, so the full
  string scores as `mozilla`: a `User-agent: Mozilla` group was consulted and a
  `User-agent: Googlebot` one never could be.
- **`--resume` re-queued URLs it had never asked for.** Filtered and blocked
  URLs were recorded as failures, indistinguishable from a fetch that failed, so
  every resumed run dispatched the whole filtered set again and spent the budget
  that should have fetched pages. They are now recorded as not-attempted.
- **A pipe did not get the per-result output the README promises.** With
  animation off the render callable was never called, so `protor scrape … | tee
  log` wrote a header, one aggregate and a path — which URLs failed appeared
  nowhere.
- **An unencodable character could still crash the CLI.** `safe()` substituted
  the glyphs the module uses and returned everything else unchanged, so a `♠`
  from a scraped page, or an `é` on an ASCII terminal, still raised at the
  write. Separately, `rich.console.Console` has no `errors` parameter and a
  `Table` never passes its cells through `print`, so a model name or page title
  in a cell raised from the middle of a render and took the error report with
  it. `safe()` is now total and the console's stream is wrapped.
- **`PROTOR_NO_LIVE=0` disabled live rendering** while `CI=0` did not — the same
  question answered two ways, and the first is never what setting a variable to
  0 meant.
- **`protor analyze -o analysis` landed in `~/Downloads/protor/analysis`,**
  because the default was the literal string `"analysis"` and the handler could
  not tell it from an explicit relative path.
- **`--backend local` and `--backend compat` were rejected by argparse** for
  names `create_backend` has always accepted; the two lists had drifted.

- **`protor update --check` reported nothing on an editable install.** It
  returned before the version check, so it printed a refusal to install and never
  said whether an update existed — in a dev tree, which is where someone is most
  likely to run it. The editable check now refuses only the *install*.
- **`--max-pages` was documented as issuing "at most 10 requests".** It is a
  ceiling over *pages*: a 502 is retried up to `MAX_RETRIES` before the page is
  recorded as failed, so one page attempt can cost three wire requests. That is
  deliberate, and the wording now says so.
- **`max_targets` held only because nothing can skip.** The ceiling counts
  dispatched pages through `stats.total`, which a skipped URL would not advance.
  Nothing can skip, because the parser yields same-host links only — an
  invariant rather than luck, and now pinned by a test on the server's request
  log.

- **One oversized block could bypass the Markdown character budget.** The budget
  is charged by `_Lines.append`, and a code block's body plus a list item's
  continuation lines were added with `list.extend`, which skips it. A single
  `<pre>` holding a minified bundle produced 151,897 characters against a 40,000
  cap — and, since the budget was never exceeded as far as the truncation check
  was concerned, no `[truncated]` marker was written either, so a caller could
  not tell the page had been cut.

### Performance

Measured before and after; the benchmark suite guards the ratio.

| | before | after |
|---|---|---|
| Page referencing one unreachable `<script src>` | 15.53 s | 0.00 s |
| Site index write, 2 000 pages | 199 MiB peak | 0.58 MiB |
| Cache open, 2 000 entries | 17.9 ms | 9.6 ms |
| Crawler checkpoint write, 40 k pages | 29.8 ms × ~8 000 | 0.17 ms × ~20 |
| `parse_html`, 173 KiB page | 159.5 ms | 75.6 ms |
| `clean_soup` | 50.5 ms | 5.8 ms |
| Live progress render | O(n²) | 10 Hz |
| LLM output | render pass per token | coalesced at ~15 Hz |

Two candidate optimisations were measured and **rejected**: `asyncio.to_thread`
around parsing (20% slower under the GIL) and `DELETE … RETURNING` for the crawl
queue (43% slower than the two statements it replaced).

### Internal

- `mypy` runs in **strict** mode and is clean. Turning it on immediately found a
  variance bug the loose settings could not see: a CLI helper returned
  manifests-or-dicts while its caller was annotated to accept only dicts.
- Tests: 505 → 1063. End-to-end coverage added for `--cache`, `--schema`
  extraction and `crawl --resume` against real HTTP servers, each mutation-checked.
- Dead code removed: `theme.simple_panel`, `runtimes.detect_runtime`, a
  byte-identical duplicate `list_models`, and `OllamaUnavailableError` (nothing
  raised it, yet `cli.py` had a handler for it and `analyze()`'s `Raises`
  section documented it).
- `tests/test_docs.py` fails the build if the README and the runtime registry
  disagree.


- Corrected comments and docstrings that did not describe the code: the
  retry-per-run semantics in `requeue_failed`, the durability window implied by
  deferred commits, `_probe`'s fallback rule, the legacy cache index format, and
  a `netguard` test whose `in (True, False)` assertion could not fail. Where a
  docstring described behaviour that did not exist, the behaviour was changed or
  the claim narrowed — not left standing.

## [v2.8.0] — 2026-10-02

### New Features

- **Support for any local model runtime.** protor was Ollama-only in practice —
  the CLI hardcoded `analyze_with_ollama` and had no `--backend` flag at all,
  even though `analyze()` already accepted a backend. Now:
  - `--backend` on `analyze`, `run` and `models`, accepting `ollama`,
    `llamacpp`, `lmstudio`, `vllm`, `localai`, `jan`, `openai-compatible`,
    `openai` and `anthropic`, plus friendly aliases (`llama.cpp`, `llama-cpp`,
    `LM-Studio`, `vLLM`, `local-ai`).
  - `--base-url` and `--api-key` to point at a runtime anywhere, or one started
    with authentication enabled.
  - New `protor runtimes` command: probes each runtime's port and prints a
    table of what is running, with the command to start the ones that are not.
  - `protor models --backend <runtime>` lists models from any runtime and
    explains how to start it when it is down.
- New `protor.runtimes` module: one `Runtime` record per runtime (default URL,
  health path, env vars, start hint, docs link), shared by the backends, the
  CLI and the error messages so a runtime is described in exactly one place.
- Per-runtime environment overrides: `OLLAMA_HOST`, `LLAMA_CPP_URL`,
  `LMSTUDIO_URL`, `VLLM_URL`, `LOCALAI_URL`, `JAN_URL`, with matching
  `*_API_KEY` variables for authenticated servers.

### Design

- **One OpenAI-compatible backend, not five.** llama.cpp, LM Studio, vLLM,
  LocalAI and Jan all implement the same `POST /v1/chat/completions` SSE
  contract, so they share a single `OpenAICompatBackend` and differ only by URL.
  Ollama keeps its native newline-delimited backend.
- Backends now use `requests` instead of vendor SDKs, so pointing protor at a
  local runtime no longer implies installing an optional cloud dependency.
  `openai` and `anthropic` remain optional extras.
- Streaming moved into one tested `_iter_sse_text`, which handles SSE framing,
  bare JSON lines, byte chunks, `data: [DONE]`, comment lines, usage-only final
  chunks, and skips `reasoning_content` so thinking traces are not shown as the
  answer.

### Changed

- **`protor crawl` starts fresh unless you pass `--resume`.** The queue database
  is opened on every run, so a previous crawl's rows used to decide what the next
  one did: running `protor crawl https://example.com` twice issued *zero* requests
  the second time and reported success. A plain crawl now clears the queue and
  visited rows first and says so; saved pages and manifests are untouched. Only
  the crawl state is discarded, and by SQL rather than by deleting the file, so
  the path stays stable and the WAL sidecars stay consistent.
- `--resume` also retries the pages the previous run failed on. They sat in
  `visited`, so nothing could re-admit them and a 502 that had since healed was
  a permanent failure; each is retried once per run, not once per rediscovery.
- `analyze` accepts `api_key`, and reports the backend's friendly display name
  ("LM Studio") rather than the raw flag value.
- Unavailable runtimes raise `RuntimeUnavailableError`, carrying the URL and the
  exact command to start it. `OllamaUnavailableError` is now a subclass of it,
  so existing handling keeps working.
- `protor models` and `analyze` normalise `--backend` aliases before argparse
  validates them; previously the factory accepted `llama.cpp` but the CLI
  rejected it.
- Every `LLMBackend` implements `list_models`, so model listing is no longer
  Ollama-specific.

### Fixed

- `ModelInfo.modified` truncated the OpenAI-compatible `created` field to 10
  characters, rendering the raw epoch (`1750000000`) as a date. Epoch values are
  now converted; ISO strings still work.
- Model listings showed an em dash for runtimes that don't report a size, rather
  than a bogus number.
- Auto-detection probes each distinct URL once, so llama.cpp and LocalAI
  sharing a port does not double-probe.

### Tests

- 353 → 441 tests. New `tests/test_runtimes.py` (30) and
  `tests/test_llm_backends.py` (61), including SSE edge cases, alias handling,
  probe semantics and per-runtime error messages. Coverage 86%.

## [v2.7.0] — 2026-10-02

A review pass over the whole pipeline, driven by measurements rather than
inspection. Every change below was reproduced with a benchmark or a failing
test first.

### Fixed — output quality

- **Streamed LLM output was being corrupted.** Analysis chunks were printed with
  rich's default `markup=True`, so a Markdown link `[docs](url)` rendered as
  `(url)`, `` `[code]` `` vanished, and an unbalanced `[/tag]` raised
  `MarkupError` — losing the entire report *after* the model had been paid for.
  Now printed with `markup=False, highlight=False`, so the terminal matches the
  saved file byte for byte.
- **Analysis silently dropped sites.** `_prepare_context` used a flat 1,500-char
  preview plus a global 8,000-char cut, so 10 sites sent only 5 while the header
  still claimed "Sites: 10". The budget is now split evenly across sites.
- **Markdown shredded paragraphs.** `<p>Protor is a <strong>fast</strong>
  scraper…</p>` became five separate lines. Inline content is now folded into a
  single line, with emphasis, links, and code preserved.
- **Adjacent lists merged.** A `<ul>` followed by an `<ol>` had no blank line
  between them, so the ordered list rendered as part of the bullet list.
- **Document order was violated.** Loose inline content was emitted before
  nested blocks, hoisting trailing links above earlier headings.
- **JavaScript files overwrote each other.** Two origins serving `vendor.js`
  collapsed to one file while the manifest still listed both URLs. Names now
  carry a short URL hash when the basename is taken.
- **Crawled pages overwrote each other.** `/docs/a.html` and `/blog/a.html`
  both wrote `a.html` and `a.manifest.json`. Filenames are now derived from the
  full path (`docs-a.html`).
- **`--max-pages` was not a real limit.** The ceiling counted only successes, so
  `--max-pages 10` issued 14 requests. Failures and blocked pages now count.
- **Checkpoints fired before any work**, and again every round while the count
  was unchanged (`scraped % n == 0` is true at zero).
- **`protor scrape` followed by `protor analyze` failed.** `analyze` defaulted
  to `data/sites_index.json` while `scrape` writes to `~/Downloads/protor`.
  The defaults now agree.
- **`run` silently ignored `--format`**, and lacked `--format`, `--prompt`, and
  `--prompt-file`, which `analyze` had. Both now share one flag helper.
- **`--prompt-file` with a bad path raised an uncaught traceback.**
- **Off-domain URLs vanished** with no stat and no log line; they are now
  recorded as `skipped`.
- **The crawl progress bar was one cell per page**, so `--max-pages 500`
  rendered a 500-character bar. Now scaled to a fixed width.
- **Cached pages reported `0 B`** and rendered as `—` in the results table.
- **`Accept-Encoding` advertised brotli** without the dependency installed, so
  servers that honoured it returned bodies aiohttp could not decode. aiohttp now
  negotiates on its own.

### Fixed — performance

- **The noise filter ran 3× per page** (once in `parse_soup`, once in
  `_extract_text`, once in `soup_to_markdown`). Now exactly once:
  **87 ms → 44 ms** per 14 KB page.
- **The rate limiter did not rate limit.** It read a timestamp, slept, then
  wrote the new one, so every concurrent task to one domain read the same stale
  value and fired together: 8 requests with a 0.5 s delay completed in 0.50 s
  instead of 3.5 s. Each waiter now reserves its own slot under a per-domain
  lock; different domains still never block each other.
- **`page_delay` slept the whole crawl loop** after every completion round,
  costing 3.7× throughput (2.21 s vs 0.68 s for 24 pages). Removed; the
  crawler now uses the per-domain rate limiter like the scraper does.
- **The HTTP cache rewrote its entire index on every page**, bodies included —
  quadratic. 60 pages of 50 KB wrote ~172 MB and blocked the event loop for
  15 ms per page. Bodies now live in one file per URL and the index is written
  once per run: **15.4 ms → 0.10 ms** per put. Caching is now opt-in
  (`--cache`) instead of always on.
- **robots.txt was fetched once per concurrent page** because the cache was
  checked before the await: 6 concurrent checks meant 6 requests. Now
  single-flight, so one crawl fetches it once.
- **The crawl queue fsynced three times per page** on the event loop. Now WAL
  journal, batched commits, and in-memory counters instead of `COUNT(*)` per
  admission check: **765 µs → 47 µs** per page.
- **Schema extraction re-parsed the HTML** with lxml even though the engine
  already held the tree. Added `extract_from_soup`.
- **The noise regex was recompiled per tag.** Hoisted to module scope.

### Changed

- `scrape_site_async` is now a thin wrapper over `CrawlEngine`. It was a second
  copy of the pipeline that had already drifted: it ignored `block_ads`, skipped
  the robots check, wrote a hard-coded `manifest.json`, and dropped `timeout`.
- The engine accepts an optional `session`, so callers holding a connection pool
  no longer open a second one.
- `page_delay` removed from `CrawlEngine`; use `rate_limiter`.
- Dropped the `aioresponses` dev dependency — 0.7.9 (latest) is incompatible
  with aiohttp 3.14. Tests use purpose-built fakes in `tests/conftest.py`.

### Tests

- **344 passing, zero warnings** (was 317 with 4 warnings). Coverage 85%.
- **Four tests were vacuous**: `AsyncMock` used as a *sync* context manager never
  entered its body, so `async with session.get(...)` never ran and the tests
  passed through the `except Exception` fallback. Rewritten against real async
  context-manager doubles — which immediately exposed that nothing had been
  verifying robots.txt rules at all.
- New `tests/test_regressions.py` pins each defect fixed above.
- Removed tests that asserted removed behaviour (`to_dict` carrying bodies).

### Removed

- Dead config: `SCALING_ENABLED` was never read; `ANALYSIS_TIMEOUT` and
  `OLLAMA_CHECK_TIMEOUT` were hardcoded at their use sites and are now the single
  source of truth.

## [v2.6.0] — 2026-07-08

### New Features
- Add Markdown output via `--markdown` flag — clean, readable conversion with noise removal
- Add crawl checkpoint/resume — `--resume` picks up where a crawl left off
- Add content filtering — `--content-filter` strips ads, nav, footer, and boilerplate
- Add SQLite-backed crawl queue — BFS ordering with persistent visited tracking
- Add User-Agent rotation — randomizes UA per request from a pool of 15 real browsers
- Add schema-based extraction — `--schema schema.json` extracts structured JSON from HTML
- Add domain/ad blocking — `--block-ads` blocks 100+ known ad/tracker domains out of the box
- Add hook system — `before_fetch`, `after_fetch`, `before_parse`, `after_parse` callbacks
- Add auto-scaling concurrency — `--auto-scale` dynamically adjusts workers based on latency
- Add `extract` command for standalone schema-based extraction from saved HTML files

### Open Source
- Add CONTRIBUTING.md with development setup, testing, and contribution guidelines
- Add GitHub issue templates (bug report + feature request)
- Add CODE_OF_CONDUCT.md
- Fix license format to SPDX expression in pyproject.toml
- Add Python 3.13 to classifiers

### Test Improvements
- Add 72 new tests for markdown, blocklist, and extractor modules (210 → 282 total)
- Increase test coverage from 69% to 79%

### Code Quality
- Fix all ruff lint errors and format issues
- Clean up unused imports across all test files

## [v2.5.0] — 2026-04-04

### New Features
- Add multi-LLM backend support: OpenAI and Anthropic alongside Ollama
- Add `llm_backends.py` module with abstract `LLMBackend` class and concrete implementations
- Add factory function `create_backend()` for easy backend switching

### Test Improvements
- Add 97 new tests (113 → 210 total), covering formatters, HTTP cache, robots.txt, rate limiter, LLM backends, scraper internals, and CLI handlers
- Increase test coverage from 61% to 84%
- Fix all pre-existing test failures and mock path issues

### Code Quality
- Fix all ruff lint issues (89 auto-fixed + 8 manual)
- Add proper exception chaining (`raise ... from exc`) throughout codebase
- Remove duplicate code (ROBOTS_PATCH, RATE_LIMIT_DELAY)
- Sort all import blocks, remove unused imports, add trailing newlines
- Replace ambiguous variable names, remove empty TYPE_CHECKING blocks

### Bug Fixes
- Fix duplicate optional dependency entries in pyproject.toml
- Fix `test_integration.py` mock paths for refactored CLI imports
- Fix `test_analyzer.py` to use `_stream_backend` instead of removed `_stream`
- Fix end-to-end test to use actual async `scrape_site_async` API

## [v2.4.0] — 2026-04-03

### New Features
- Add progress callbacks for library users
- Add custom headers support for auth tokens
- Add HTTP caching to avoid re-fetching unchanged pages
- Add custom prompts via CLI `--prompt` flag or file `--prompt-file`

## [v2.2.0] — 2026-04-03

### New Features
- Add `update` command to check for and install updates from PyPI
- Add per-domain rate limiting with configurable politeness delays
- Add robots.txt support — blocks scraping of disallowed URLs
- Add `python -m protor` support via `__main__.py`
- Add URL validation at CLI entry points with helpful error messages

### Improvements
- Migrate all hardcoded values to `config.py` (crawler delay, concurrency, headers, Ollama base, max data chars)
- Improve `human_bytes` precision using float division
- Update CI workflow: replace flake8/black with ruff, add mypy type-checking
- Support Python 3.11, 3.12, 3.13 in CI matrix

### Bug Fixes
- Fix pre-existing test failures in `test_scraper.py`, `test_crawler.py`, `test_cli.py`, `test_analyzer.py`
- Fix `_abort` test that incorrectly expected `SystemExit` when `sys.exit` was mocked
- Add missing `mock_ollama_response` fixture to `conftest.py`
- Register `asyncio` marker in pytest configuration

## [v2.1.0] — 2026-04-03

### New Features
- Add `update` command to check for and install updates from PyPI
- Support `--check` flag for version info only
- Support `-y` flag to skip confirmation prompt
- Detect editable installs and show appropriate update instructions

## [v2.0.0] — 2026-04-02

### Breaking Changes
- Migrate from curl-based scraping to async aiohttp
- Python 3.11+ required (was 3.8+)
- Default output directory changed to `~/Downloads/protor`

### New Features
- Add crawler with BFS algorithm and live progress display
- Add typed dataclasses: `SiteManifest`, `SiteMetadata`, `AnalysisResult`
- Implement proper exception hierarchy with typed errors
- Add Rich theme system for consistent CLI output
- Refactor CLI to use subcommand pattern with proper error handling
- Add `version` command
- Add concurrency control with `--concurrency` flag

### Security Fixes
- Enable SSL certificate verification (was disabled)
- Add prompt injection protection in analyzer prompts

### Bug Fixes
- Fix asyncio.gather re-await pattern in scraper orchestrator
- Fix silent error swallowing in crawler — errors now logged with details
- Fix cross-platform path resolution using `Path.home()`
- Fix error count reporting in scraper summary

### Improvements
- Modernize pyproject.toml with full tool configurations (ruff, mypy, pytest, coverage)
- Update test suite with proper fixtures and async support
- Add Dockerfile for containerized deployment
- Add comprehensive docstrings across all modules
- Cleaner, more maintainable codebase with proper type hints

### Internal
- Remove `requirements-dev.txt` — use `pyproject.toml` optional deps
- Add `protor/__init__.py` with version detection
- Add `protor/exceptions.py` for typed errors
- Add `protor/models.py` for data classes
- Add `protor/theme.py` for Rich console theming
