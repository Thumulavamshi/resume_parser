from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import tempfile
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import re
import copy
import PyPDF2
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Resume Parser API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.1-8b-instant"

PROMPT = '''Extract all possible candidate information from the following resume. You MUST return ONLY a valid JSON object with no additional text, explanations, or markdown formatting.

The JSON must have exactly two top-level keys:
- personal_info: an object with keys name, email, phone, linkedin, github, website (always include these, use "not found" if missing)
- other_info: an object with these exact keys (always include, use "not found" if missing): education, skills, experience, projects. All other extracted fields (achievements, publications, certifications, positions, etc.) can be included as appropriate.

Resume:
"""
{resume_text}
"""

Return only the JSON object (no markdown, no explanations):'''

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {str(e)}")

def get_val(d, keys):
    for k in keys:
        v = d.get("personal_info", {}).get(k) or d.get(k)
        if v and v != "not found":
            # Handle different data types
            if isinstance(v, str):
                return v.strip() if v.strip() else "not found"
            elif isinstance(v, dict):
                # If it's a dict, try to get a meaningful string representation
                return str(v) if v else "not found"
            elif isinstance(v, list):
                # If it's a list, join the elements or return first element
                return ", ".join(str(item) for item in v) if v else "not found"
            else:
                return str(v) if v else "not found"
    return "not found"

def ensure_key(d, key):
    v = d.get(key)
    if v is None or (isinstance(v, str) and v.strip().lower() == "not found"):
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        if key == "courses":
            return [
                {"name": k, "issuer": val}
                for k, val in v.items() if val and (isinstance(val, str) and val.strip().lower() != "not found")
            ]
        if key == "certifications":
            return [
                {"name": k, "issuer": val}
                for k, val in v.items() if val and (isinstance(val, str) and val.strip().lower() != "not found")
            ]
        if all(isinstance(val, str) for val in v.values()):
            return [{"key": k, "value": val} for k, val in v.items()]
        return [dict({"key": k}, **val) if isinstance(val, dict) else {"key": k, "value": val} for k, val in v.items()]
    if isinstance(v, list):
        return v
    return [v]

def normalize_resume_data(data: dict) -> dict:
    """Normalize the extracted resume data"""
    personal_info = {
        "name": get_val(data, ["name"]),
        "email": get_val(data, ["email"]),
        "phone": get_val(data, ["phone"]),
        "linkedin": get_val(data, ["linkedin", "linkedin_url", "linkedin_profile"]),
        "github": get_val(data, ["github", "github_url", "github_profile"]),
        "website": get_val(data, ["website", "portfolio", "personal_website", "web"])
    }

    other_info = dict(data)
    for k in ["personal_info", "name", "email", "phone", "linkedin", "github", "website"]:
        other_info.pop(k, None)

    if "other_info" in other_info and isinstance(other_info["other_info"], dict):
        for k, v in other_info["other_info"].items():
            other_info[k] = v
        other_info.pop("other_info", None)

    other_info_final = {}

    # Education normalization
    edu_val = ensure_key(other_info, "education")
    norm_edu = []
    for entry in edu_val:
        if isinstance(entry, dict):
            inst = entry.get("institution") or entry.get("institute") or entry.get("key") or ""
            degree = entry.get("degree", "")
            grade = entry.get("cgpa") or entry.get("score") or entry.get("gpa") or entry.get("grade") or ""
            level = ""
            if "b.tech" in degree.lower() or "bachelor" in degree.lower():
                level = "ug"
            elif "intermediate" in degree.lower() or "12" in degree or "junior" in inst.lower():
                level = "12th"
            elif "matriculation" in degree.lower() or "10" in degree or "concept school" in inst.lower():
                level = "10th"
            norm_edu.append({k: v for k, v in [("institution", inst), ("degree", degree), ("grade", grade), ("level", level)] if v})
        else:
            norm_edu.append(entry)
    other_info_final["education"] = norm_edu

    # Experience
    other_info_final["experience"] = ensure_key(other_info, "experience")

    # Projects normalization
    proj_val = ensure_key(other_info, "projects")
    norm_proj = []
    for entry in proj_val:
        if isinstance(entry, dict):
            title = entry.get("name") or entry.get("key") or entry.get("title") or ""
            desc = entry.get("description") or entry.get("Description") or []
            if isinstance(desc, str):
                desc = [desc]
            for k, v in entry.items():
                if k.lower() not in ["name", "key", "title", "description", "Description"] and v:
                    if isinstance(v, list):
                        desc.extend([str(i) for i in v])
                    else:
                        desc.append(str(v))
            norm_proj.append({"title": title, "description": desc})
        else:
            norm_proj.append(entry)
    other_info_final["projects"] = norm_proj

    # Extra info
    extra_info = {}
    for k, v in other_info.items():
        if k not in ["education", "experience", "projects", "skills", "achievements", "publications", "certifications", "positions"]:
            extra_info[k] = v

    # Skills flattening
    skills_val = ensure_key(other_info, "skills")
    flat_skills = []
    if isinstance(skills_val, dict):
        for group in skills_val.values():
            if isinstance(group, list):
                flat_skills.extend(group)
            elif isinstance(group, str):
                flat_skills.append(group)
    elif isinstance(skills_val, list):
        for group in skills_val:
            if isinstance(group, dict):
                for v in group.values():
                    if isinstance(v, list):
                        flat_skills.extend(v)
                    elif isinstance(v, str):
                        flat_skills.append(v)
            elif isinstance(group, str):
                flat_skills.append(group)
    if flat_skills:
        extra_info["skills"] = flat_skills

    for k in ["achievements", "publications", "certifications", "positions"]:
        v = ensure_key(other_info, k)
        if v:
            extra_info[k] = v
    
    if extra_info:
        other_info_final["extra_info"] = copy.deepcopy(extra_info)

    return {
        "personal_info": personal_info,
        "other_info": other_info_final
    }

@app.get("/")
async def root():
    return {"message": "Resume Parser API", "status": "running"}

@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    """
    Parse a resume file (PDF or TXT) and return structured JSON data
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    # Validate file size (max 10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size too large. Maximum size is 10MB")
    
    # Validate file type
    allowed_types = ["application/pdf", "text/plain"]
    allowed_extensions = [".pdf", ".txt"]
    
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    
    # Additional validation by file extension
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="File must have .pdf or .txt extension")
    
    logger.info(f"Processing file: {file.filename}, size: {file.size}, type: {file.content_type}")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Extract text based on file type
        if file.content_type == "application/pdf":
            resume_text = extract_text_from_pdf(file_content)
        else:
            resume_text = file_content.decode('utf-8')
        
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        # Process with LLM
        prompt = PROMPT.format(resume_text=resume_text)
        llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL)
        response = llm.invoke(prompt)
        
        # Parse JSON response
        try:
            data = json.loads(response.content)
        except Exception:
            # Try to extract JSON substring with better regex
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'```\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*?\})', response.content, re.DOTALL)
            
            if json_match:
                try:
                    json_str = json_match.group(1).strip()
                    # Clean up common JSON formatting issues
                    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                    json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                    data = json.loads(json_str)
                except Exception as e:
                    # If still fails, create a fallback response with the raw text
                    logger.error(f"JSON parsing error: {e}")
                    logger.error(f"Raw response: {response.content}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"LLM returned invalid JSON. Raw response: {response.content[:500]}..."
                    )
            else:
                # If no JSON found, create a fallback response
                raise HTTPException(
                    status_code=500, 
                    detail=f"No JSON found in LLM output. Raw response: {response.content[:500]}..."
                )
        
        # Normalize the data
        final_result = normalize_resume_data(data)
        
        return final_result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "groq_key_configured": bool(GROQ_API_KEY)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)