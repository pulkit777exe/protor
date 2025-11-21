import os
import time
import json
import subprocess
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from rich.console import Console
from rich.panel import Panel
from rich import box
from .utils import safe_filename, save_json, timestamp

console = Console()

def fetch_with_curl(url: str, timeout: int = 30) -> tuple[str, bool]:
    """Fetch URL content using curl. Returns (content, success)"""
    try:
        result = subprocess.run(
            ["curl", "-sL", "--max-time", str(timeout), "-A", "Mozilla/5.0", url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout + 5
        )
        success = result.returncode == 0 and len(result.stdout) > 0
        return result.stdout, success
    except Exception as e:
        console.print(f"[grey50]⟡ Summoning failed for {url}: {e}[/grey50]")
        return "", False

def extract_metadata(soup: BeautifulSoup) -> dict:
    """Extract metadata from HTML"""
    metadata = {
        "title": "",
        "description": "",
        "keywords": [],
        "author": "",
        "og_tags": {}
    }
    
    if soup.title:
        metadata["title"] = soup.title.string.strip()
    
    for meta in soup.find_all("meta"):
        name = meta.get("name", "").lower()
        prop = meta.get("property", "").lower()
        content = meta.get("content", "")
        
        if name == "description":
            metadata["description"] = content
        elif name == "keywords":
            metadata["keywords"] = [k.strip() for k in content.split(",")]
        elif name == "author":
            metadata["author"] = content
        elif prop.startswith("og:"):
            metadata["og_tags"][prop] = content
    
    return metadata

def extract_js_links(html: str, base_url: str) -> list[str]:
    """Extract JavaScript file links from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    js_links = []
    
    for script in soup.find_all("script", src=True):
        src = script["src"]
        full_url = urljoin(base_url, src)
        js_links.append(full_url)
    
    return list(set(js_links))


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract internal links from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    base_domain = urlparse(base_url).netloc
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        
        # Only keep internal links (same domain) and http(s)
        if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
            # Remove fragments
            clean_url = full_url.split("#")[0]
            if clean_url != base_url:
                links.append(clean_url)
    
    return list(set(links))

def extract_text_content(html: str) -> str:
    """Extract clean text content from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
    
    text = soup.get_text(separator="\n")
    
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text[:10000]

def download_file(url: str, dest_path: str, timeout: int = 15) -> bool:
    """Download a file to destination path"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        r = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0"
        })
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        console.print(f"[grey50]  ✗ Artifact lost: {url}[/grey50]")
    return False

def scrape_website(url: str, output_dir: str = "data", download_js: bool = True, timeout: int = 30) -> dict:
    """Scrape a single website"""
    console.print(f"[grey74]⟡ Summoning:[/grey74] [grey50]{url}[/grey50]")
    
    parsed = urlparse(url)
    site_dir = os.path.join(output_dir, safe_filename(parsed.netloc))
    os.makedirs(site_dir, exist_ok=True)

    html, success = fetch_with_curl(url, timeout)
    if not success or not html:
        console.print(f"[grey50]  ✗ Soul escaped: {url}[/grey50]")
        return None

    html_path = os.path.join(site_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    soup = BeautifulSoup(html, "html.parser")
    metadata = extract_metadata(soup)
    text_content = extract_text_content(html)
    
    js_downloaded = []
    if download_js:
        js_links = extract_js_links(html, url)
        if js_links:
            console.print(f"[grey50]  ⟡ {len(js_links)} artifacts discovered[/grey50]")
            js_dir = os.path.join(site_dir, "js")
            os.makedirs(js_dir, exist_ok=True)
            
            for idx, js_url in enumerate(js_links[:10]):
                fname = safe_filename(os.path.basename(js_url)) or f"script_{idx}.js"
                dest = os.path.join(js_dir, fname)
                if download_file(js_url, dest):
                    js_downloaded.append(js_url)
                time.sleep(0.3)

    manifest = {
        "url": url,
        "domain": parsed.netloc,
        "html_file": html_path,
        "metadata": metadata,
        "text_content": text_content,
        "js_files": js_downloaded,
        "js_count": len(js_downloaded),
        "timestamp": timestamp(),
        "success": True
    }

    # save manifest
    save_json(manifest, os.path.join(site_dir, "manifest.json"))
    console.print(f"[grey74]  ✓ Soul captured:[/grey74] [grey50]{parsed.netloc}[/grey50]")
    
    return manifest

def scrape_multiple(urls: list[str], output_dir: str = "data", download_js: bool = True, timeout: int = 30) -> str:
    """Scrape multiple websites and create an index"""
    console.print()
    console.print(Panel(
        f"[bold grey93]⸸ Beginning the Harvest ⸸[/bold grey93]\n"
        f"[grey74]Targets:[/grey74] [bright_white]{len(urls)}[/bright_white]\n"
        f"[grey74]Crypt:[/grey74] [grey50]{output_dir}[/grey50]",
        box=box.DOUBLE_EDGE,
        border_style="grey35",
        style="on grey7"
    ))
    console.print()
    
    manifests = []
    
    for i, url in enumerate(urls, 1):
        console.print(f"[grey50]━━━[/grey50] [grey74][{i}/{len(urls)}][/grey74] [grey50]━━━[/grey50]")
        manifest = scrape_website(url, output_dir, download_js, timeout)
        if manifest:
            manifests.append(manifest)
        time.sleep(1)
    
    index_file = os.path.join(output_dir, "sites_index.json")
    save_json(manifests, index_file)
    
    console.print()
    console.print(Panel(
        f"[bold grey93]⸸ Harvest Complete ⸸[/bold grey93]\n"
        f"[grey74]Claimed:[/grey74] [bright_white]{len(manifests)}[/bright_white][grey50]/[/grey50][grey74]{len(urls)}[/grey74]\n"
        f"[grey74]Tome:[/grey74] [grey50]{index_file}[/grey50]",
        box=box.DOUBLE_EDGE,
        border_style="grey35",
        style="on grey7"
    ))
    console.print()
    
    return index_file