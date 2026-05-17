import os
import json
import requests
from vinni.rag import retrieve
from dotenv import load_dotenv

load_dotenv()


# ── TOOL IMPLEMENTATIONS ────────────────────────────────────────

def get_contact_info() -> str:
    return json.dumps({
        "name": "Vidhya Sree Narayanappa",
        "email": "vidhyasreenarayanappa@gmail.com",
        "phone": "(857) 230-4249",
        "location": "Boston, MA",
        "linkedin": "https://www.linkedin.com/in/vidhya-sree-n/",
        "github": "https://github.com/VidhyaSree-N",
        "portfolio": "coming soon"
    })

def get_current_role() -> str:
    return json.dumps({
        "title": "Full Stack Software Engineer",
        "company": "RoWorks AI",
        "location": "Boston, MA",
        "period": "January 2024 to Present",
        "description": "Primary engineer building a physical AI and robotics automation platform end to end — iOS app, web dashboard, backend microservices, and AWS infrastructure.",
        "tech_stack": ["Swift", "SwiftUI", "React", "Node.js", "Python", "AWS Lambda", "EC2", "PostgreSQL", "MongoDB", "Docker"]
    })

def get_skills(category: str = "all") -> str:
    skills = {
        "ios": ["Swift", "SwiftUI", "ARKit", "RealityKit", "LiDAR", "AVFoundation", "CoreData", "APNs", "Xcode", "TestFlight"],
        "frontend": ["React", "Next.js", "TypeScript", "JavaScript", "HTML5", "CSS3", "Tailwind CSS", "Redux", "Three.js", "WebGL"],
        "backend": ["Node.js", "Express", "Python", "FastAPI", "Java", "Spring Boot", ".NET Core", "REST APIs", "microservices"],
        "cloud": ["AWS Lambda", "EC2", "S3", "SQS", "SNS", "Amplify", "ALB", "Docker", "Nginx", "CI/CD", "GitHub Actions"],
        "databases": ["PostgreSQL", "MongoDB", "MySQL", "Oracle", "SQL Server"],
        "ai_ml": ["OpenAI API", "Anthropic Claude API", "Gemini API", "RAG pipelines", "LangChain", "Qdrant", "PyTorch", "TensorFlow", "Scikit-learn", "GroundingDINO", "SAMURAI"]
    }

    if category == "all":
        return json.dumps(skills)
    elif category in skills:
        return json.dumps({category: skills[category]})
    else:
        return json.dumps({"error": f"Unknown category: {category}. Available: {list(skills.keys())}"})
    
def list_projects() -> str:
    projects = [
        {
            "name": "RoWorks iOS App",
            "type": "iOS",
            "company": "RoWorks AI",
            "tech": ["Swift", "SwiftUI", "ARKit", "RealityKit", "LiDAR", "AWS S3", "APNs"],
            "description": "Primary data collection iOS app for manufacturing clients. AR scanning with 3D bounding box placement, LiDAR capture, and multi-file upload pipeline to AWS S3."
        },
        {
            "name": "RoWorks Web Platform",
            "type": "Web",
            "company": "RoWorks AI",
            "tech": ["React", "Node.js", "PostgreSQL", "MongoDB", "Docker", "Nginx", "AWS"],
            "description": "Full customer and integrator marketplace dashboard with role-based auth, DocuSign NDA signing, robot catalog, and Gemini AI chat assistant."
        },
        {
            "name": "RoWorks Backend Microservices",
            "type": "Backend",
            "company": "RoWorks AI",
            "tech": ["Node.js", "Python", "PostgreSQL", "MongoDB", "AWS Lambda", "EC2", "SQS", "SNS"],
            "description": "Six microservices architecture with two-stage AI perception pipeline using GroundingDINO and SAMURAI on EC2 with GPU support."
        },
        {
            "name": "Boostlet.js",
            "type": "Open Source Research",
            "company": "Machine Psychology Lab, UMass Boston",
            "tech": ["JavaScript", "WebGL", "OpenSeadragon"],
            "description": "JavaScript plugin system for scientific image visualization published at IEEE VIS 2024."
        },
        {
            "name": "Gamified Research Platform",
            "type": "Full Stack Research",
            "company": "Machine Psychology Lab, UMass Boston",
            "tech": ["React", "Node.js", "Python", "Scikit-learn", "PostgreSQL"],
            "description": "Full-stack gamified participation tracking platform with ML behavioral classifier."
        },
        {
            "name": "Vinni AI",
            "type": "AI Agent",
            "company": "Personal Project",
            "tech": ["Python", "OpenAI", "LangChain", "Qdrant", "Whisper", "TTS"],
            "description": "Personal AI voice and text agent that answers questions about Vidhya using RAG and OpenAI tool calling."
        }
    ]
    return json.dumps(projects)

def search_profile(query: str) -> str:
    chunks = retrieve(query, top_k=3)
    return json.dumps({"results": chunks})

def fetch_github_projects() -> str:
    try:
        url = "https://api.github.com/users/VidhyaSree-N/repos"
        response = requests.get(url, params={"sort": "updated", "per_page": 10})
        repos = response.json()

        projects = []
        for repo in repos:
            if not repo.get("fork"):  # skip forked repos
                projects.append({
                    "name": repo["name"],
                    "description": repo.get("description", "No description"),
                    "language": repo.get("language", "Unknown"),
                    "url": repo["html_url"],
                    "updated": repo["updated_at"][:10]
                })

        return json.dumps({"github_projects": projects})
    except Exception as e:
        return json.dumps({"error": f"Could not fetch GitHub projects: {str(e)}"})

# ── TOOL DEFINITIONS (what GPT reads) ──────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_contact_info",
            "description": "Returns Vidhya's contact information including email, phone number, LinkedIn, GitHub, and location. Use this when someone asks how to contact Vidhya, wants her email, phone, LinkedIn, or GitHub links.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_role",
            "description": "Returns Vidhya's current job and role details including company, title, responsibilities, and tech stack. Use this when someone asks where Vidhya works, what her current job is, or what she does at RoWorks AI.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_skills",
            "description": "Returns Vidhya's technical skills optionally filtered by category. Use this when someone asks about Vidhya's skills, technologies, or what she knows. Categories are: ios, frontend, backend, cloud, databases, ai_ml, or all.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Skill category to filter by. Options: ios, frontend, backend, cloud, databases, ai_ml, all",
                        "enum": ["ios", "frontend", "backend", "cloud", "databases", "ai_ml", "all"]
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "Returns a structured list of all projects Vidhya has built including name, type, tech stack, and description. Use this when someone asks about Vidhya's projects, portfolio, or what she has built.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_profile",
            "description": "Searches Vidhya's detailed profile using semantic search and returns the most relevant information. Use this for detailed or specific questions about Vidhya's experience, background, achievements, education, or anything not covered by other tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information about Vidhya"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_github_projects",
            "description": "Fetches Vidhya's latest public projects directly from her GitHub profile in real time. Use this when someone asks about her GitHub, wants to see her code, or asks for her latest projects.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# ── TOOL ROUTER ─────────────────────────────────────────────────
# This is what gets called when GPT picks a tool
def execute_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name == "get_contact_info":
        return get_contact_info()
    elif tool_name == "get_current_role":
        return get_current_role()
    elif tool_name == "get_skills":
        category = tool_args.get("category", "all")
        return get_skills(category)
    elif tool_name == "list_projects":
        return list_projects()
    elif tool_name == "search_profile":
        query = tool_args.get("query", "")
        return search_profile(query)
    elif tool_name == "fetch_github_projects":
        return fetch_github_projects()
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
