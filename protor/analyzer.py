import os
import json
import requests
from typing import Dict, List
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.table import Table
from rich import box
from .utils import save_json, timestamp

console = Console()

ANALYSIS_PROMPTS = {
    "general": """
You are an AI analyst. Analyze the scraped website data and provide:
1. **Overview**: What is this website about?
2. **Key Content**: Main topics and themes
3. **Technical Stack**: Technologies detected
4. **Data Insights**: Interesting patterns or information
5. **Recommendations**: Potential use cases or improvements

Be concise and insightful.
""",
    "technical": """
You are a technical analyst. Focus on:
1. **Tech Stack**: Frontend/backend technologies detected
2. **JavaScript Analysis**: Frameworks, libraries, APIs used
3. **Performance**: Page structure and optimization opportunities
4. **Security**: Potential concerns or best practices
5. **Architecture**: Overall technical approach
""",
    "content": """
You are a content analyst. Focus on:
1. **Content Quality**: Writing style and clarity
2. **SEO Elements**: Titles, descriptions, keywords
3. **Structure**: Information hierarchy and organization
4. **Engagement**: Call-to-actions and user journey
5. **Audience**: Target demographic and tone
""",
    "seo": """
You are an SEO specialist. Analyze:
1. **Meta Tags**: Title, description, keywords quality
2. **Content Structure**: Headers, semantic HTML
3. **Technical SEO**: Page speed indicators, mobile-friendliness
4. **Improvements**: Specific SEO recommendations
5. **Competitive Edge**: Unique value propositions
"""
}

def check_ollama_connection() -> bool:
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def list_ollama_models():
    """List available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                table = Table(
                    show_header=True,
                    header_style="bold grey93 on grey11",
                    box=box.DOUBLE_EDGE,
                    border_style="grey35"
                )
                table.add_column("Oracle", style="grey74")
                table.add_column("Size", style="grey50", justify="right")
                
                for model in models:
                    name = model.get("name", "unknown")
                    size = model.get("size", 0) / (1024**3)
                    table.add_row(name, f"{size:.1f} GB")
                
                console.print()
                console.print(Panel(
                    table,
                    title="[bold grey93]⟪ Available Oracles ⟫[/bold grey93]",
                    box=box.DOUBLE_EDGE,
                    border_style="grey35",
                    style="on grey7"
                ))
                console.print()
            else:
                console.print()
                console.print(Panel(
                    "[grey74]No oracles found in the realm.[/grey74]\n\n"
                    "[grey50]Summon an oracle with:[/grey50]\n"
                    "[grey74]ollama pull <model-name>[/grey74]",
                    title="[bold grey93]⚠ Empty Sanctuary ⚠[/bold grey93]",
                    box=box.DOUBLE_EDGE,
                    border_style="grey35",
                    style="on grey7"
                ))
                console.print()
        else:
            console.print("[grey50]⟡ Cannot reach the spirit realm[/grey50]")
    except Exception as e:
        console.print()
        console.print(Panel(
            f"[grey74]Connection failed:[/grey74] [grey50]{e}[/grey50]\n\n"
            "[grey74]Ensure Ollama serves:[/grey74]\n"
            "[grey50]ollama serve[/grey50]",
            title="[bold grey93]⚠ Oracle Unreachable ⚠[/bold grey93]",
            box=box.DOUBLE_EDGE,
            border_style="grey35",
            style="on grey7"
        ))
        console.print()

def prepare_analysis_data(data: List[Dict], max_chars: int = 6000) -> str:
    """Prepare scraped data for analysis"""
    summary = []
    
    for idx, site in enumerate(data, 1):
        site_info = f"""
## Site {idx}: {site.get('domain', 'Unknown')}
- **URL**: {site.get('url', 'N/A')}
- **Title**: {site.get('metadata', {}).get('title', 'N/A')}
- **Description**: {site.get('metadata', {}).get('description', 'N/A')}
- **JavaScript Files**: {site.get('js_count', 0)}

### Content Preview:
{site.get('text_content', '')[:1000]}
"""
        summary.append(site_info)
    
    full_text = "\n---\n".join(summary)
    
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[Content truncated...]"
    
    return full_text

def stream_ollama_response(model: str, prompt: str) -> str:
    """Stream response from Ollama API"""
    endpoint = "http://localhost:11434/api/generate"
    
    try:
        response = requests.post(
            endpoint,
            json={
                "model": model,
                "prompt": prompt,
                "stream": True
            },
            stream=True,
            timeout=300
        )
        
        if response.status_code != 200:
            return f"Error: Ollama returned status {response.status_code}"
        
        full_response = []
        
        console.print()
        console.print("[grey50]" + "─" * 60 + "[/grey50]")
        console.print("[grey74 italic]⟡ The oracle speaks...[/grey74 italic]")
        console.print("[grey50]" + "─" * 60 + "[/grey50]")
        console.print()
        
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        text = chunk["response"]
                        console.print(text, end="", style="grey74")
                        full_response.append(text)
                    
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        
        console.print()
        console.print()
        console.print("[grey50]" + "─" * 60 + "[/grey50]")
        console.print()
        
        return "".join(full_response)
        
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Try a smaller model or reduce data size."
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"

def analyze_with_ollama(data: List[Dict], model: str = "llama3", focus: str = "general", output_dir: str = "analysis"):
    """Analyze scraped data using Ollama"""
    
    if not check_ollama_connection():
        console.print()
        console.print(Panel(
            "[bold grey93]⚠ Oracle Unavailable ⚠[/bold grey93]\n\n"
            "[grey74]Cannot establish connection to Ollama[/grey74]\n\n"
            "[grey50]Ensure the oracle serves:[/grey50]\n"
            "[grey74]1. Start Ollama:[/grey74] [grey50]ollama serve[/grey50]\n"
            "[grey74]2. Summon a model:[/grey74] [grey50]ollama pull llama3[/grey50]",
            box=box.DOUBLE_EDGE,
            border_style="grey35",
            style="on grey7"
        ))
        console.print()
        return
    
    console.print()
    console.print(Panel(
        f"[bold grey93]⸸ Preparing the Divination ⸸[/bold grey93]\n"
        f"[grey74]Oracle:[/grey74] [bright_white]{model}[/bright_white]\n"
        f"[grey74]Focus:[/grey74] [grey74]{focus}[/grey74]\n"
        f"[grey74]Souls to divine:[/grey74] [bright_white]{len(data)}[/bright_white]\n\n"
        f"[grey50 italic]The ritual may take a moment...[/grey50 italic]",
        box=box.DOUBLE_EDGE,
        border_style="grey35",
        style="on grey7"
    ))
    
    data_summary = prepare_analysis_data(data)
    analysis_prompt = ANALYSIS_PROMPTS.get(focus, ANALYSIS_PROMPTS["general"])
    
    full_prompt = f"""{analysis_prompt}

## Scraped Website Data:
{data_summary}

Provide your analysis in well-formatted Markdown:
"""
    
    result = stream_ollama_response(model, full_prompt)
    
    if result.startswith("Error:"):
        console.print()
        console.print(Panel(
            f"[grey74]{result}[/grey74]",
            title="[bold grey93]⚠ Divination Failed ⚠[/bold grey93]",
            box=box.DOUBLE_EDGE,
            border_style="grey35",
            style="on grey7"
        ))
        console.print()
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    analysis_data = {
        "model": model,
        "focus": focus,
        "timestamp": timestamp(),
        "sites_analyzed": len(data),
        "analysis": result
    }
    save_json(analysis_data, os.path.join(output_dir, "analysis.json"))
    
    md_content = f"""# Website Analysis Report

**Generated**: {timestamp()}  
**Model**: {model}  
**Focus**: {focus}  
**Sites Analyzed**: {len(data)}

---

{result}

---

*Generated by Protor - AI-powered web scraping & analysis*
"""
    
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(md_content)
    
    console.print()
    console.print(Panel(
        f"[bold grey93]⸸ Prophecy Inscribed ⸸[/bold grey93]\n"
        f"[grey74]Tome:[/grey74] [grey50]{output_dir}/README.md[/grey50]\n"
        f"[grey74]Scroll:[/grey74] [grey50]{output_dir}/analysis.json[/grey50]",
        box=box.DOUBLE_EDGE,
        border_style="grey35",
        style="on grey7"
    ))
    console.print()