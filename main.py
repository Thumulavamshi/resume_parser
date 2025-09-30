from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
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

app = FastAPI(title="Resume Parser & Question Generation API", version="1.0.0")

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

# Supported models for question generation with fallback
SUPPORTED_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.2-3b-preview", 
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

PROMPT = '''Extract all possible candidate information from the following resume. You MUST return ONLY a valid JSON object with no additional text, explanations, or markdown formatting.

The JSON must have exactly two top-level keys:
- personal_info: an object with keys name, email, phone, linkedin, github, website (always include these, use "not found" if missing)
- other_info: an object with these exact keys (always include, use "not found" if missing): education, skills, experience, projects. All other extracted fields (achievements, publications, certifications, positions, etc.) can be included as appropriate.

Resume:
"""
{resume_text}
"""

Return only the JSON object (no markdown, no explanations):'''

# Pydantic models for question generation
class ResumeInput(BaseModel):
    personal_info: Dict[str, Any]
    other_info: Dict[str, Any]
    technology: str = "reactjs"  # or "nodejs"

class Question(BaseModel):
    id: int
    question: str
    difficulty: str
    category: str
    expected_topics: List[str]

class QuestionsOutput(BaseModel):
    questions: List[Question]
    technology: str
    candidate_name: str

# Answer Scoring Models
class CandidateInfo(BaseModel):
    name: str
    technology: str

class InterviewQuestion(BaseModel):
    question_id: int
    question: str
    difficulty: str
    category: str
    expected_topics: List[str]
    answer: str
    time_taken: int
    max_time_allowed: int

class ScoringInput(BaseModel):
    candidate_info: CandidateInfo
    interview_data: List[InterviewQuestion]

class QuestionScore(BaseModel):
    question_id: int
    question: str
    difficulty: str
    category: str
    candidate_answer: str
    time_taken: int
    max_time_allowed: int
    content_score: float  # 0-10
    total_score: float  # 0-10 (same as content_score)
    feedback: str
    strengths: List[str]
    weaknesses: List[str]
    key_points_covered: List[str]
    key_points_missed: List[str]

class DifficultyBreakdown(BaseModel):
    easy: Dict[str, float]
    medium: Dict[str, float]
    hard: Dict[str, float]

class FinalScore(BaseModel):
    content_score: float  # 0-100
    overall_score: float  # 0-100 (same as content_score)

class ScoringOutput(BaseModel):
    candidate_name: str
    technology: str
    total_questions: int
    questions_attempted: int
    question_scores: List[QuestionScore]
    difficulty_breakdown: DifficultyBreakdown
    final_score: FinalScore
    overall_feedback: str
    recommendation: str
    strengths_summary: List[str]
    areas_for_improvement: List[str]

QUESTION_GENERATION_PROMPT = """You are an expert technical interviewer specializing in {technology}. Generate exactly 6 interview questions based on the candidate's resume.

**ABSOLUTE REQUIREMENTS (MUST FOLLOW):**
✓ EXACTLY 2 questions with difficulty: "easy"
✓ EXACTLY 2 questions with difficulty: "medium"  
✓ EXACTLY 2 questions with difficulty: "hard"
✓ All 6 questions MUST focus on {technology}
✓ Reference candidate's {technology} experience/projects from resume when possible
✓ Use realistic scenarios based on their background

**Question Strategy:**
- EASY (2 questions): Core concepts, syntax, basic understanding
- MEDIUM (2 questions): Practical implementation, debugging, best practices, specific to their project experience
- HARD (2 questions): Architecture decisions, complex scenarios, system design related to their work

**Candidate Background:**
{resume_summary}

**Your Task:** Create 6 {technology} interview questions that test different skill levels. If the candidate has {technology} projects or experience mentioned above, create questions that relate to those specific projects or technologies they've used.

**EXACT OUTPUT FORMAT (JSON only, no other text):**
{{
  "questions": [
    {{
      "id": 1,
      "question": "Easy {technology} question here?",
      "difficulty": "easy",
      "category": "fundamentals",
      "expected_topics": ["{technology_lower}", "basics"]
    }},
    {{
      "id": 2, 
      "question": "Second easy {technology} question here?",
      "difficulty": "easy",
      "category": "fundamentals", 
      "expected_topics": ["{technology_lower}", "concepts"]
    }},
    {{
      "id": 3,
      "question": "Medium {technology} question here?", 
      "difficulty": "medium",
      "category": "best-practices",
      "expected_topics": ["{technology_lower}", "implementation"]
    }},
    {{
      "id": 4,
      "question": "Second medium {technology} question here?",
      "difficulty": "medium", 
      "category": "debugging",
      "expected_topics": ["{technology_lower}", "problem-solving"]
    }},
    {{
      "id": 5,
      "question": "Hard {technology} question here?",
      "difficulty": "hard",
      "category": "system-design", 
      "expected_topics": ["{technology_lower}", "architecture"]
    }},
    {{
      "id": 6,
      "question": "Second hard {technology} question here?",
      "difficulty": "hard",
      "category": "design-patterns",
      "expected_topics": ["{technology_lower}", "advanced"]
    }}
  ]
}}

Remember: Return ONLY valid JSON. Count carefully - exactly 2 easy, 2 medium, 2 hard questions about {technology}."""

# Answer Scoring Prompts
EVALUATION_PROMPT = """You are an expert technical interviewer evaluating a candidate's answer for a {technology} position.

**Question #{question_id} ({difficulty} - {category}):**
{question}

**Expected Topics:** {expected_topics}

**Candidate's Answer:**
{answer}

**Evaluation Instructions:**
1. Assess the technical accuracy and depth of the answer
2. Check if expected topics are covered
3. Evaluate clarity and structure of the response
4. Consider the difficulty level in your scoring
5. Be strict but fair - empty or very poor answers should get 0-2 scores

**Scoring Criteria:**
- **Content Score (0-10):**
  - 9-10: Excellent, comprehensive, accurate
  - 7-8: Good understanding, minor gaps
  - 5-6: Basic understanding, missing key points
  - 3-4: Partial/incorrect understanding
  - 0-2: No answer or completely wrong

**Output JSON format:**
{{
  "content_score": 7.5,
  "feedback": "Detailed evaluation of the answer...",
  "strengths": ["specific strength 1", "specific strength 2"],
  "weaknesses": ["specific weakness 1", "specific weakness 2"],
  "key_points_covered": ["point1", "point2"],
  "key_points_missed": ["point1", "point2"]
}}

Evaluate now:"""

OVERALL_SUMMARY_PROMPT = """Based on the complete interview evaluation, provide an overall assessment.

**Candidate:** {candidate_name}
**Technology:** {technology}
**Overall Score:** {overall_score}/100

**Performance Breakdown:**
{performance_breakdown}

**Difficulty-wise Performance:**
- Easy Questions: {easy_avg}/10
- Medium Questions: {medium_avg}/10
- Hard Questions: {hard_avg}/10

Provide a JSON response:
{{
  "overall_feedback": "2-3 sentences summarizing overall performance",
  "recommendation": "strong-hire/hire/conditional-hire/no-hire",
  "strengths_summary": ["key strength 1", "key strength 2", "key strength 3"],
  "areas_for_improvement": ["area 1", "area 2", "area 3"]
}}

**Recommendation Guidelines:**
- strong-hire: 80-100, excellent across all difficulties
- hire: 65-79, good performance with minor gaps
- conditional-hire: 50-64, basic understanding, needs improvement
- no-hire: below 50, insufficient knowledge

Generate the summary:"""

def is_answer_valid(answer: str) -> bool:
    """Check if answer is meaningful (not gibberish)"""
    if not answer or len(answer.strip()) < 5:
        return False
    
    # Check for gibberish (repeated characters, no spaces, etc.)
    words = answer.strip().split()
    if len(words) < 2:
        return False
    
    # Check if it's mostly random characters
    alpha_count = sum(c.isalpha() for c in answer)
    if alpha_count < len(answer) * 0.7:
        return False
    
    return True

def extract_json_from_response(content: str) -> dict:
    """Extract JSON from LLM response, handling various formats"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Try to find any JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
    
    return None

def create_resume_summary(resume_data: Dict[str, Any], technology: str = "") -> str:
    """Create a focused summary of the resume highlighting relevant tech experience"""
    summary_parts = []
    
    # Personal info
    personal = resume_data.get("personal_info", {})
    name = personal.get("name", "Unknown")
    summary_parts.append(f"Candidate: {name}")
    
    # Education
    other_info = resume_data.get("other_info", {})
    education = other_info.get("education", [])
    if education:
        edu_summary = []
        for edu in education[:2]:
            if isinstance(edu, dict):
                degree = edu.get("degree", "")
                institution = edu.get("institution", "")
                if degree or institution:
                    edu_summary.append(f"{degree} from {institution}")
        if edu_summary:
            summary_parts.append(f"Education: {'; '.join(edu_summary)}")
    
    # Experience - Focus on tech-relevant roles
    experience = other_info.get("experience", [])
    if experience:
        tech_keywords = ["react", "node", "javascript", "frontend", "backend", "full-stack", "developer", "engineer"]
        relevant_exp = []
        
        for exp in experience[:3]:
            if isinstance(exp, dict):
                role_raw = exp.get("role", exp.get("title", ""))
                role = str(role_raw).lower() if role_raw else ""
                company_raw = exp.get("company", "")
                company = str(company_raw) if company_raw else ""
                description_raw = exp.get("description", "")
                description = str(description_raw).lower() if description_raw else ""
                
                # Check if experience is tech-related
                is_relevant = any(keyword in role or keyword in description for keyword in tech_keywords)
                
                if is_relevant or not relevant_exp:  # Include first experience even if not explicitly tech
                    role_display = str(role_raw) if role_raw else ""
                    if role_display:
                        relevant_exp.append(f"{role_display} at {company}")
        
        if relevant_exp:
            summary_parts.append(f"Relevant Experience: {'; '.join(relevant_exp)}")
    
    # Projects - Highlight React/Node.js projects
    projects = other_info.get("projects", [])
    if projects:
        tech_projects = []
        other_projects = []
        
        for proj in projects:
            if isinstance(proj, dict):
                title_raw = proj.get("title", proj.get("name", ""))
                title = str(title_raw) if title_raw else ""
                description_raw = proj.get("description", "")
                description = str(description_raw).lower() if description_raw else ""
                technologies = proj.get("technologies", [])
                
                if isinstance(technologies, list):
                    # Convert all items to string before joining
                    tech_text = " ".join(str(tech) for tech in technologies).lower()
                else:
                    tech_text = str(technologies).lower() if technologies else ""
                
                # Check if project uses relevant technology
                search_text = f"{description} {tech_text}".lower()
                tech_match = any(tech in search_text for tech in ["react", "node", "javascript", "js", "frontend", "backend"])
                
                if tech_match:
                    tech_projects.append(f"{title} (uses relevant technology)")
                elif title:
                    other_projects.append(title)
        
        # Prioritize tech-relevant projects
        display_projects = tech_projects[:2] + other_projects[:2]
        if display_projects:
            summary_parts.append(f"Key Projects: {'; '.join(display_projects[:3])}")
    
    # Skills - Highlight relevant technical skills
    extra_info = other_info.get("extra_info", {})
    skills = extra_info.get("skills", [])
    if skills and isinstance(skills, list):
        # Prioritize relevant skills
        relevant_skills = []
        other_skills = []
        
        priority_skills = ["react", "node", "javascript", "typescript", "html", "css", "redux", "express", "mongodb", "sql"]
        
        for skill in skills:
            skill_str = str(skill) if skill else ""
            skill_lower = skill_str.lower()
            if any(priority in skill_lower for priority in priority_skills):
                relevant_skills.append(skill_str)
            else:
                other_skills.append(skill_str)
        
        # Show relevant skills first, then others
        display_skills = relevant_skills[:8] + other_skills[:4]
        if display_skills:
            summary_parts.append(f"Technical Skills: {', '.join(display_skills[:10])}")
    
    return "\n".join(summary_parts)

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
    return {
        "message": "Resume Parser, Question Generation & Answer Scoring API", 
        "status": "running",
        "endpoints": ["/parse-resume", "/generate-questions", "/score-answers", "/health"]
    }

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

@app.post("/generate-questions", response_model=QuestionsOutput)
async def generate_questions(resume_input: ResumeInput):
    """
    Generate 6 interview questions based on resume and technology stack
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    try:
        # Create resume summary with technology focus
        resume_data = {
            "personal_info": resume_input.personal_info,
            "other_info": resume_input.other_info
        }
        resume_summary = create_resume_summary(resume_data, resume_input.technology)
        
        # Format technology name
        tech_display = "React.js" if resume_input.technology.lower() == "reactjs" else "Node.js"
        
        # Generate prompt
        prompt = QUESTION_GENERATION_PROMPT.format(
            resume_summary=resume_summary,
            technology=tech_display,
            technology_lower=tech_display.lower()
        )
        
        # Call LLM with fallback models
        response = None
        last_error = None
        
        for model_name in SUPPORTED_MODELS:
            try:
                llm = ChatGroq(api_key=GROQ_API_KEY, model=model_name, temperature=0.7)
                response = llm.invoke(prompt)
                break  # Success, exit the loop
            except Exception as e:
                last_error = e
                if "decommissioned" in str(e) or "model" in str(e).lower():
                    continue  # Try next model
                else:
                    raise  # Re-raise if it's not a model issue
        
        if response is None:
            raise HTTPException(
                status_code=500,
                detail=f"All models failed. Last error: {str(last_error)}"
            )
        
        # Parse JSON response
        try:
            # Try direct JSON parse
            questions_data = json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if json_match:
                questions_data = json.loads(json_match.group(1))
            else:
                # Try to find any JSON object
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    questions_data = json.loads(json_match.group(0))
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="Could not parse questions from LLM response"
                    )
        
        # Validate we have 6 questions
        questions = questions_data.get("questions", [])
        if len(questions) != 6:
            raise HTTPException(
                status_code=500,
                detail=f"Expected 6 questions, got {len(questions)}"
            )
        
        # Count difficulty levels and validate distribution
        difficulty_count = {"easy": 0, "medium": 0, "hard": 0}
        for q in questions:
            diff = q.get("difficulty", "").lower()
            if diff in difficulty_count:
                difficulty_count[diff] += 1
        
        # Log the difficulty distribution for debugging
        expected_distribution = {"easy": 2, "medium": 2, "hard": 2}
        if difficulty_count != expected_distribution:
            logger.warning(f"Expected {expected_distribution}, got {difficulty_count}")
        
        # Get candidate name
        candidate_name = resume_input.personal_info.get("name", "Unknown")
        
        return QuestionsOutput(
            questions=questions,
            technology=tech_display,
            candidate_name=candidate_name
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating questions: {str(e)}"
        )

@app.post("/score-answers", response_model=ScoringOutput)
async def score_answers(scoring_input: ScoringInput):
    """
    Score candidate's interview answers with detailed evaluation
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    if not scoring_input.interview_data:
        raise HTTPException(status_code=400, detail="No interview data provided")
    
    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0.3)
        question_scores = []
        questions_attempted = 0
        
        # Evaluate each question
        for qa in scoring_input.interview_data:
            # Check if answer is valid
            answer_valid = is_answer_valid(qa.answer)
            
            if not answer_valid:
                # Auto-score for invalid/gibberish answers
                question_score = QuestionScore(
                    question_id=qa.question_id,
                    question=qa.question,
                    difficulty=qa.difficulty,
                    category=qa.category,
                    candidate_answer=qa.answer,
                    time_taken=qa.time_taken,
                    max_time_allowed=qa.max_time_allowed,
                    content_score=0.0,
                    total_score=0.0,  # Same as content_score
                    feedback="Invalid or incomplete answer provided. The response appears to be gibberish or too short to evaluate.",
                    strengths=[],
                    weaknesses=["No meaningful answer provided", "Appears to be random text"],
                    key_points_covered=[],
                    key_points_missed=qa.expected_topics
                )
                question_scores.append(question_score)
                continue
            
            questions_attempted += 1
            
            # Generate evaluation prompt
            prompt = EVALUATION_PROMPT.format(
                technology=scoring_input.candidate_info.technology,
                question_id=qa.question_id,
                difficulty=qa.difficulty,
                category=qa.category,
                question=qa.question,
                expected_topics=", ".join(qa.expected_topics),
                answer=qa.answer
            )
            
            # Get evaluation from LLM
            response = llm.invoke(prompt)
            eval_data = extract_json_from_response(response.content)
            
            if not eval_data:
                # Fallback evaluation
                eval_data = {
                    "content_score": 5.0,
                    "feedback": "Could not parse evaluation properly. Manual review recommended.",
                    "strengths": [],
                    "weaknesses": ["Evaluation parsing failed"],
                    "key_points_covered": [],
                    "key_points_missed": qa.expected_topics
                }
            
            # Calculate scores (content only)
            content_score = float(eval_data.get("content_score", 5.0))
            
            # Total score is same as content score
            total_score = content_score
            
            question_score = QuestionScore(
                question_id=qa.question_id,
                question=qa.question,
                difficulty=qa.difficulty,
                category=qa.category,
                candidate_answer=qa.answer,
                time_taken=qa.time_taken,
                max_time_allowed=qa.max_time_allowed,
                content_score=round(content_score, 2),
                total_score=round(total_score, 2),
                feedback=eval_data.get("feedback", ""),
                strengths=eval_data.get("strengths", []),
                weaknesses=eval_data.get("weaknesses", []),
                key_points_covered=eval_data.get("key_points_covered", []),
                key_points_missed=eval_data.get("key_points_missed", [])
            )
            question_scores.append(question_score)
        
        # Calculate difficulty breakdown
        difficulty_scores = {"easy": [], "medium": [], "hard": []}
        
        for score in question_scores:
            diff = score.difficulty.lower()
            if diff in difficulty_scores:
                difficulty_scores[diff].append(score.content_score)
        
        difficulty_breakdown = DifficultyBreakdown(
            easy={
                "avg_content_score": round(sum(difficulty_scores["easy"]) / len(difficulty_scores["easy"]), 2) if difficulty_scores["easy"] else 0.0,
                "count": len(difficulty_scores["easy"])
            },
            medium={
                "avg_content_score": round(sum(difficulty_scores["medium"]) / len(difficulty_scores["medium"]), 2) if difficulty_scores["medium"] else 0.0,
                "count": len(difficulty_scores["medium"])
            },
            hard={
                "avg_content_score": round(sum(difficulty_scores["hard"]) / len(difficulty_scores["hard"]), 2) if difficulty_scores["hard"] else 0.0,
                "count": len(difficulty_scores["hard"])
            }
        )
        
        # Calculate final scores
        all_content_scores = [s.content_score for s in question_scores]
        
        content_score_avg = sum(all_content_scores) / len(all_content_scores)
        
        # Convert to 0-100 scale
        content_score_100 = round((content_score_avg / 10) * 100, 2)
        
        # Overall score is same as content score
        overall_score = content_score_100
        
        final_score = FinalScore(
            content_score=content_score_100,
            overall_score=overall_score
        )
        
        # Generate overall summary
        performance_breakdown = "\n".join([
            f"Q{s.question_id} ({s.difficulty}): {s.total_score}/10"
            for s in question_scores
        ])
        
        summary_prompt = OVERALL_SUMMARY_PROMPT.format(
            candidate_name=scoring_input.candidate_info.name,
            technology=scoring_input.candidate_info.technology,
            overall_score=overall_score,
            performance_breakdown=performance_breakdown,
            easy_avg=difficulty_breakdown.easy["avg_content_score"],
            medium_avg=difficulty_breakdown.medium["avg_content_score"],
            hard_avg=difficulty_breakdown.hard["avg_content_score"]
        )
        
        summary_response = llm.invoke(summary_prompt)
        summary_data = extract_json_from_response(summary_response.content)
        
        if not summary_data:
            summary_data = {
                "overall_feedback": f"Candidate scored {overall_score}/100 overall.",
                "recommendation": "conditional-hire" if overall_score >= 50 else "no-hire",
                "strengths_summary": ["Requires manual review"],
                "areas_for_improvement": ["Requires manual review"]
            }
        
        return ScoringOutput(
            candidate_name=scoring_input.candidate_info.name,
            technology=scoring_input.candidate_info.technology,
            total_questions=len(scoring_input.interview_data),
            questions_attempted=questions_attempted,
            question_scores=question_scores,
            difficulty_breakdown=difficulty_breakdown,
            final_score=final_score,
            overall_feedback=summary_data.get("overall_feedback", ""),
            recommendation=summary_data.get("recommendation", "conditional-hire"),
            strengths_summary=summary_data.get("strengths_summary", []),
            areas_for_improvement=summary_data.get("areas_for_improvement", [])
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error scoring answers: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "groq_key_configured": bool(GROQ_API_KEY),
        "models_available": SUPPORTED_MODELS[0]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)