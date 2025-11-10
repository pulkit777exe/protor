import os
import json
import requests
from typing import Dict, List
from .utils import save_json, timestamp

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
                print("Available Ollama models:")
                for model in models:
                    name = model.get("name", "unknown")
                    size = model.get("size", 0) / (1024**3) 
                    print(f"   • {name} ({size:.1f} GB)")
            else:
                print("No models found. Install models with: ollama pull <model-name>")
        else:
            print("Could not connect to Ollama")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")

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
        print("\n" + "="*60)
        
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        text = chunk["response"]
                        print(text, end="", flush=True)
                        full_response.append(text)
                    
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        
        print("\n" + "="*60 + "\n")
        return "".join(full_response)
        
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Try a smaller model or reduce data size."
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"

def analyze_with_ollama(data: List[Dict], model: str = "llama3", focus: str = "general", output_dir: str = "analysis"):
    """Analyze scraped data using Ollama"""
    
    if not check_ollama_connection():
        print("Cannot connect to Ollama!")
        print("Make sure Ollama is running:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Pull a model: ollama pull llama3")
        return
    
    data_summary = prepare_analysis_data(data)
    analysis_prompt = ANALYSIS_PROMPTS.get(focus, ANALYSIS_PROMPTS["general"])
    
    full_prompt = f"""{analysis_prompt}

## Scraped Website Data:
{data_summary}

Provide your analysis in well-formatted Markdown:
"""
    
    print(f"Analysis Focus: {focus}")
    print(f"Analyzing {len(data)} website(s) with {model}...")
    print("This may take a minute...\n")
    
    result = stream_ollama_response(model, full_prompt)
    
    if result.startswith("Error:"):
        print(f"{result}")
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
    
    print(f"\n Analysis saved to:")
    print(f"   • {output_dir}/README.md")
    print(f"   • {output_dir}/analysis.json")
