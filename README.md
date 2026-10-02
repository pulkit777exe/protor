# protor

> scrape websites. analyze with ai. no bs.

a cli tool that actually works. scrapes web content with async aiohttp, feeds it to a local llm, gets insights. works with **ollama, llama.cpp, LM Studio, vLLM, LocalAI, Jan**, or the OpenAI/Anthropic APIs. that's it.

## why this exists

because paying for web scraping apis is kinda mid when you can just use aiohttp and a local llm. also because sometimes you need to analyze a bunch of sites and doing it manually is literally painful.

## what you need

- python 3.11+
- a local model runtime (any of the ones below), or an OpenAI/Anthropic key

### pick your runtime

protor talks to any local runtime. llama.cpp, LM Studio, vLLM, LocalAI and Jan
all speak the same OpenAI-compatible API, so they work out of the box; ollama
has its own native API and is supported directly.

```bash
# see what's running right now
protor runtimes

# then list the models it has loaded
protor models --backend <runtime>
```

| `--backend` | Default URL | Start it with |
|---|---|---|
| `ollama` (default) | `http://localhost:11434` | `ollama serve` |
| `llamacpp` (alias `llama.cpp`) | `http://localhost:8080` | `llama-server -m model.gguf` |
| `lmstudio` (alias `lm-studio`) | `http://localhost:1234` | `lms server start` |
| `vllm` | `http://localhost:8000` | `vllm serve <model>` |
| `localai` | `http://localhost:8081` | `localai run` |
| `jan` | `http://localhost:1337` | enable the local server in Jan |
| `openai-compatible` | — | any OpenAI-compatible server |
| `openai` / `anthropic` | — | set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |

override the URL with `--base-url`, or set a per-runtime environment variable
(`OLLAMA_HOST`, `LLAMA_CPP_URL`, `LMSTUDIO_URL`, `VLLM_URL`, `LOCALAI_URL`,
`JAN_URL`). If a runtime was started with authentication on, pass `--api-key` or
set the matching `*_API_KEY` variable — no header is sent when there's no token.

### get ollama set up (the easy default)

```bash
# grab some models
ollama pull llama3
ollama pull mistral
ollama pull codellama

# start the server
ollama serve
```

## install

### from pypi (recommended)

```bash
pip install protor
```

### from source

```bash
git clone https://github.com/pulkit777exe/protor.git
cd protor
pip install -e .
```

## how to use

### see what models you have

```bash
# what's running locally?
protor runtimes

# models from the default runtime (ollama)
protor models

# models from a different runtime
protor models --backend lmstudio
protor models --backend vllm
protor models --backend llama.cpp
```

### scrape stuff

```bash
# one site
protor scrape https://example.com

# multiple sites
protor scrape https://example.com https://another-site.com

# skip the js files if you want
protor scrape https://example.com --no-js

# custom settings
protor scrape https://example.com --output my_data --timeout 60 --concurrency 3
```

### crawl a whole site

```bash
# crawl up to 10 pages
protor crawl https://example.com

# deeper crawl
protor crawl https://example.com --max-pages 50
```

### analyze what you scraped

```bash
# general vibes check
protor analyze

# tech stack deep dive
protor analyze --focus technical --model codellama

# seo audit
protor analyze --focus seo --model mistral

# content analysis
protor analyze --focus content

# use a different local runtime
protor analyze --backend lmstudio --model granite-4-micro
protor analyze --backend llamacpp --model ./qwen3-8b-q4.gguf
protor analyze --backend vllm --model Qwen/Qwen3-8B

# or a hosted API
protor analyze --backend openai --model gpt-4o

# runtime listening somewhere unusual
protor analyze --backend vllm --base-url http://gpu-box:8000
```

### do both at once (recommended)

```bash
# basic usage
protor run https://example.com

# with options
protor run https://example.com https://another.com --model llama3 --focus technical

# go crazy
protor run https://site1.com https://site2.com https://site3.com \
  --model mistral \
  --focus seo \
  --no-js

# scrape with ollama and it just works; or pick a runtime
protor run https://example.com --backend lmstudio --model granite-4-micro
```

### check your version

```bash
protor version
```

### keep it up to date

```bash
# check for updates
protor update --check

# update with confirmation
protor update

# skip the prompt
protor update -y
```

## what the focus modes do

- **general** - overall content, main themes, what the site's about
- **technical** - frameworks, tech stack, how it's built
- **content** - writing quality, structure, how readable it is
- **seo** - meta tags, optimization stuff, what needs fixing

## what you get

### after scraping

```
<output>/
├── example_com/
│   ├── index.html        # the actual html
│   ├── manifest.json     # metadata and stuff
│   └── js/               # javascript files
└── sites_index.json      # summary of everything
```

The default output directory is `~/Downloads/protor`, and `protor analyze`
looks there by default — so `protor scrape <url>` followed by `protor analyze`
just works. Pass `-o DIR` to both to change it.

When you `crawl` a multi-page site, each page gets a collision-free name
derived from its full path (`docs/guide.html` → `docs-guide.html`), so pages
with the same leaf name never overwrite each other. Downloaded scripts are
disambiguated the same way when two origins serve the same basename.

### after analysis

```
analysis/
├── analysis.md          # readable report (or .txt/.csv/.html per --format)
└── analysis.json        # raw data
```

## real examples

### quick content check

```bash
protor run https://blog.example.com --focus content
```

### technical audit

```bash
# grab everything including js
protor scrape https://webapp.example.com

# analyze the tech
protor analyze --focus technical --model codellama
```

### competitor research

```bash
# scrape competitors
protor scrape https://competitor1.com https://competitor2.com https://competitor3.com

# get seo insights
protor analyze --focus seo --model mistral
```

### batch analysis

```bash
protor run \
  https://source1.com \
  https://source2.com \
  https://source3.com \
  --model llama3
```

## when stuff breaks

### runtime not detected

```bash
# what's actually running?
protor runtimes

# list models from a specific runtime
protor models --backend <runtime>

# if it listens somewhere unusual
protor models --backend vllm --base-url http://gpu-box:8000
```

`protor runtimes` prints the exact command to start each runtime, so a
"Cannot reach vLLM at http://localhost:8000" error tells you what to run.

### ollama issues

```bash
# make sure it's running
ollama serve

# check your models
ollama list

# pull a model if needed
ollama pull llama3
```

### connection failing

```bash
# try longer timeout
protor scrape https://example.com --timeout 120

# reduce concurrency if getting blocked
protor scrape https://example.com --concurrency 2
```

### analysis taking forever

- use a smaller model
- scrape fewer sites
- use --no-js flag
- get better hardware lol

## pro tips

- always check robots.txt before scraping (be respectful)
- start with --no-js if you just need content
- codellama is best for technical analysis
- mistral is faster than llama3
- use custom output dirs for different projects
- retry logic is built-in for transient failures (429, 5xx)

## what's inside

```
protor/
├── cli.py          # command interface
├── engine.py       # the crawl loop shared by scrape + crawl (queue, links, limits)
├── fetcher.py      # http: retries, ua rotation, hooks, conditional requests
├── parser.py       # one html parse -> text, markdown, links, js refs
├── scraper.py      # batch scraping orchestrator + live table
├── crawler.py      # bfs site crawler with sqlite queue, checkpoint/resume
├── analyzer.py     # runtime-agnostic analysis + model listing
├── extractor.py    # schema-based structured data extraction
├── markdown.py     # html to clean markdown converter
├── blocklist.py    # ad/tracker domain blocking (100+ domains)
├── models.py       # typed dataclasses
├── exceptions.py   # error hierarchy
├── config.py       # centralized constants
├── llm_backends.py # ollama native + openai-compatible backends
├── runtimes.py     # local runtime registry + auto-detection
├── theme.py        # rich console theming
├── http_cache.py   # conditional http caching (opt-in via --cache)
├── robots.py       # robots.txt support (single-flight, cached)
├── rate_limiter.py # concurrency-safe per-domain rate limiting
├── updater.py      # pypi update checker
├── formatters.py   # output formatting
└── utils.py        # helper stuff
```

## contributing

see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## customize it

want different analysis prompts? edit the `_PROMPTS` dict in `protor/analyzer.py`

need different timeouts or concurrency? check `protor/config.py`

## changelog

see [CHANGELOG.md](CHANGELOG.md) for the full history.

## legal stuff

mit license. do whatever you want with it.

just don't be weird and scrape sites that explicitly say no. respect robots.txt. don't ddos anyone. you know, basic internet etiquette.

## tech stack

- ollama / llama.cpp / LM Studio / vLLM / LocalAI / Jan (local llm inference)
- beautifulsoup4 + lxml (html parsing)
- aiohttp (async http)
- rich (cli output)

---

built because web scraping shouldn't require a phd or a credit card

made with spite and caffeine

star it if it's useful idk
