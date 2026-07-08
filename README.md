# protor

> scrape websites. analyze with ai. no bs.

a cli tool that actually works. scrapes web content with async aiohttp, feeds it to your local ollama models, gets insights. that's it.

## why this exists

because paying for web scraping apis is kinda mid when you can just use aiohttp and a local llm. also because sometimes you need to analyze a bunch of sites and doing it manually is literally painful.

## what you need

- python 3.11+
- [ollama](https://ollama.ai) running locally {with model of your choice}

### get ollama set up

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
protor models
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
data/
├── example_com/
│   ├── index.html        # the actual html
│   ├── manifest.json     # metadata and stuff
│   └── js/               # javascript files
└── sites_index.json      # summary of everything
```

### after analysis

```
analysis/
├── README.md            # readable report
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
├── scraper.py      # async html + js scraper with hooks, UA rotation, auto-scaling
├── crawler.py      # bfs site crawler with checkpoint/resume
├── analyzer.py     # ollama/openai/anthropic integration
├── extractor.py    # schema-based structured data extraction
├── markdown.py     # html to clean markdown converter
├── blocklist.py    # ad/tracker domain blocking (100+ domains)
├── models.py       # typed dataclasses
├── exceptions.py   # error hierarchy
├── config.py       # centralized constants
├── llm_backends.py # multi-backend llm abstraction
├── theme.py        # rich console theming
├── http_cache.py   # conditional http caching
├── robots.py       # robots.txt support
├── rate_limiter.py # per-domain rate limiting
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

- ollama (local llm inference)
- beautifulsoup4 + lxml (html parsing)
- aiohttp (async http)
- rich (cli output)

---

built because web scraping shouldn't require a phd or a credit card

made with spite and caffeine

star it if it's useful idk
