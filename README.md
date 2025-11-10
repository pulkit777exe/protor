# protor

> scrape websites. analyze with ai. no bs.

a cli tool that actually works. scrapes web content with curl, feeds it to your local ollama models, gets insights. that's it.

## why this exists

because paying for web scraping apis is kinda mid when you can just use curl and a local llm. also because sometimes you need to analyze a bunch of sites and doing it manually is literally painful.

## what you need

- python 3.8+ (obviously)
- curl (you probably have it)
- [ollama](https://ollama.ai) running locally

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

```bash
# clone this
git clone <your-repo-url>
cd basic-llm-web-scraper

# install it
pip install -e .

# or just
pip install -r requirements.txt
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
protor scrape https://example.com --output my_data --timeout 60
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

### curl failing

```bash
# test manually
curl -sL https://example.com

# try longer timeout
protor scrape https://example.com --timeout 120
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

## what's inside

```
protor/
├── cli.py          # command interface
├── scraper.py      # does the scraping
├── analyzer.py     # talks to ollama
└── utils.py        # helper stuff
```

## customize it

want different analysis prompts? edit the `ANALYSIS_PROMPTS` in `protor/analyzer.py`

need different rate limits? check `protor/scraper.py` (0.3s between js files, 1s between sites)

## legal stuff

mit license. do whatever you want with it.

just don't be weird and scrape sites that explicitly say no. respect robots.txt. don't ddos anyone. you know, basic internet etiquette.

## tech stack

- ollama (local llm inference)
- beautifulsoup (html parsing)
- requests (http stuff)
- curl (the goat)

---

built because web scraping shouldn't require a phd or a credit card

made with spite and caffeine

star it if it's useful idk