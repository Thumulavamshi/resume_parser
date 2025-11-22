"""
RESUME-BASED NATURAL INTERVIEW SYSTEM
======================================
- Parse resume with consistent JSON structure
- Generate questions purely from resume content
- Natural interview flow (no difficulty levels)
- Answer scoring with detailed feedback
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
import json
import re
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import logging
from docx import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI(title="Resume-Based Interview System", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================
# PYDANTIC MODELS - PARSING
# ============================================

class PersonalInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None

class Education(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    grade: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    achievements: Optional[List[str]] = None

class Experience(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration: Optional[str] = None
    location: Optional[str] = None
    responsibilities: Optional[List[str]] = None
    technologies_used: Optional[List[str]] = None
    key_achievements: Optional[List[str]] = None

class Project(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None
    technologies: Optional[List[str]] = None
    key_features: Optional[List[str]] = None
    challenges_solved: Optional[List[str]] = None
    link: Optional[str] = None
    duration: Optional[str] = None

class SkillCategory(BaseModel):
    languages: Optional[List[str]] = None
    frameworks: Optional[List[str]] = None
    databases: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    cloud_platforms: Optional[List[str]] = None
    other: Optional[List[str]] = None

class Certification(BaseModel):
    name: Optional[str] = None
    issuer: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    credential_id: Optional[str] = None

class EnhancedResumeOutput(BaseModel):
    personal_info: PersonalInfo
    education: Optional[List[Education]] = None
    experience: Optional[List[Experience]] = None
    projects: Optional[List[Project]] = None
    skills: Optional[SkillCategory] = None
    certifications: Optional[List[Certification]] = None
    achievements: Optional[List[str]] = None
    publications: Optional[List[str]] = None
    languages: Optional[List[str]] = None

# ============================================
# PYDANTIC MODELS - QUESTIONS
# ============================================

class InterviewQuestion(BaseModel):
    id: int
    question: str
    category: str  # introduction/experience/project/skill/achievement
    expected_topics: List[str]
    context: Optional[str] = None

class InterviewQuestionSet(BaseModel):
    candidate_name: str
    questions: List[InterviewQuestion]
    total_questions: int

# ============================================
# PYDANTIC MODELS - SCORING
# ============================================

class CandidateInfo(BaseModel):
    name: str

class InterviewAnswer(BaseModel):
    question_id: int
    question: str
    category: str
    expected_topics: List[str]
    answer: str
    time_taken: Optional[int] = None

class ScoringInput(BaseModel):
    candidate_info: CandidateInfo
    interview_data: List[InterviewAnswer]

class QuestionScore(BaseModel):
    question_id: int
    question: str
    category: str
    candidate_answer: str
    time_taken: Optional[int] = None
    score: float  # 0-10
    feedback: str
    strengths: List[str]
    weaknesses: List[str]
    key_points_covered: List[str]
    key_points_missed: List[str]

class FinalScore(BaseModel):
    overall_score: float  # 0-100

class ScoringOutput(BaseModel):
    candidate_name: str
    total_questions: int
    questions_attempted: int
    question_scores: List[QuestionScore]
    final_score: FinalScore
    overall_feedback: str
    recommendation: str
    strengths_summary: List[str]
    areas_for_improvement: List[str]

# ============================================
# PROMPTS - PARSING
# ============================================

RESUME_PARSING_PROMPT = """Extract all information from the resume into structured JSON.

**RULES:**
1. Return ONLY valid JSON, no other text
2. Use null for missing fields
3. Extract maximum detail for projects and experience
4. Identify all technologies, tools, and skills mentioned

**JSON SCHEMA:**
{{
  "personal_info": {{
    "name": "string or null",
    "email": "string or null",
    "phone": "string or null",
    "linkedin": "string or null",
    "github": "string or null",
    "website": "string or null",
    "location": "string or null"
  }},
  "education": [
    {{
      "institution": "string or null",
      "degree": "string or null",
      "field_of_study": "string or null",
      "grade": "string or null",
      "start_date": "string or null",
      "end_date": "string or null",
      "achievements": ["achievement1"] or null
    }}
  ],
  "experience": [
    {{
      "company": "string or null",
      "role": "string or null",
      "start_date": "string or null",
      "end_date": "string or null",
      "duration": "string or null",
      "location": "string or null",
      "responsibilities": ["resp1", "resp2"] or null,
      "technologies_used": ["tech1", "tech2"] or null,
      "key_achievements": ["achievement1"] or null
    }}
  ],
  "projects": [
    {{
      "title": "string or null",
      "description": "string or null",
      "role": "Solo/Team Lead/Contributor or null",
      "technologies": ["tech1", "tech2"] or null,
      "key_features": ["feature1"] or null,
      "challenges_solved": ["challenge1"] or null,
      "link": "string or null",
      "duration": "string or null"
    }}
  ],
  "skills": {{
    "languages": ["lang1"] or null,
    "frameworks": ["framework1"] or null,
    "databases": ["db1"] or null,
    "tools": ["tool1"] or null,
    "cloud_platforms": ["platform1"] or null,
    "other": ["skill1"] or null
  }},
  "certifications": [
    {{
      "name": "string or null",
      "issuer": "string or null",
      "issue_date": "string or null",
      "expiry_date": "string or null",
      "credential_id": "string or null"
    }}
  ],
  "achievements": ["achievement1"] or null,
  "publications": ["publication1"] or null,
  "languages": ["language1"] or null
}}

**RESUME TEXT:**
\"\"\"
{resume_text}
\"\"\"

Return JSON only:"""

# ============================================
# PROMPTS - QUESTION GENERATION (IMPROVED)
# ============================================

RESUME_BASED_QUESTION_GENERATION = """You are designing a natural, realistic technical interview based purely on the candidate's resume.

**CANDIDATE RESUME:**
{resume_summary}

**INTERVIEW DESIGN PRINCIPLES:**

This should feel like a real interview where the interviewer has read the resume and is asking questions about what they see. Questions must be DIRECTLY related to what is mentioned in the resume - no generic technology questions, no difficulty levels, no predefined tech focus.

**QUESTION 1 - INTRODUCTION (MANDATORY):**
Always start with this exact introduction request:

"Tell me about yourself — your background, education, any experience you have, and projects you've worked on. Feel free to share any interests or hobbies as well."

**QUESTIONS 2-7 - RESUME-SPECIFIC QUESTIONS:**
Generate 6 questions based ONLY on what's actually in the resume:

**Question Types Based on Resume Content:**

IF the resume has **professional experience**:
  - "Can you tell me about your role at [Company Name]? What were your main responsibilities?"
  - "What was the most challenging aspect of working at [Company Name]?"
  - "What did you learn during your time at [Company Name]?"

IF the resume has **projects**:
  - "I see you worked on [Project Name]. Can you walk me through what that project was about?"
  - "What was your role in the [Project Name] project?"
  - "What challenges did you face while working on [Project Name]?"
  - "What was the most interesting part of building [Project Name]?"

IF specific **technologies/tools** are mentioned in projects or experience:
  - "You mentioned using [Technology] in your project. Why did you choose that?"
  - "How did you use [Tool/Technology] in your work?"
  - "What was your experience working with [Technology]?"

IF the resume mentions **achievements**:
  - "Can you elaborate on [Achievement mentioned]?"
  - "How did you accomplish [Achievement]?"

IF the resume has **certifications**:
  - "I see you have a certification in [Certification Name]. What motivated you to pursue that?"
  - "How have you applied what you learned from [Certification]?"

IF the resume mentions **education** but little/no experience:
  - "What subjects or areas did you focus on during your studies?"
  - "What was your favorite project or coursework during your degree?"

**CRITICAL RULES:**
1. Every question (Q2-Q7) MUST reference something SPECIFIC from the resume
2. Use actual names: project titles, company names, technologies THEY mentioned
3. DO NOT ask about technologies not mentioned in the resume
4. DO NOT use generic questions like "Explain React" unless React is in their resume
5. Questions should flow naturally like a real conversation
6. Adapt questions to the candidate's level (student/junior/senior based on resume)

**OUTPUT JSON FORMAT:**
{{
  "questions": [
    {{
      "id": 1,
      "question": "Tell me about yourself — your background, education, any experience you have, and projects you've worked on. Feel free to share any interests or hobbies as well.",
      "category": "introduction",
      "expected_topics": ["background", "education", "experience", "projects", "interests"],
      "context": "Opening introduction"
    }},
    {{
      "id": 2,
      "question": "[Specific question about something in their resume]",
      "category": "experience/project/skill/achievement/education",
      "expected_topics": ["topic1", "topic2", "topic3"],
      "context": "Asked because candidate mentioned [specific item] in resume"
    }},
    {{
      "id": 3,
      "question": "[Another specific resume-based question]",
      "category": "experience/project/skill/achievement/education",
      "expected_topics": ["topic1", "topic2"],
      "context": "Asked because candidate has [specific background/project/experience]"
    }}
    // ... continue for Q4-Q7
  ]
}}

**IMPORTANT REMINDERS:**
- Total 7 questions (1 intro + 6 resume-based)
- Every question should make the candidate feel like you actually read their resume
- Natural interview tone - conversational, not interrogative
- NO difficulty levels
- NO generic technology questions
- Questions should reflect the candidate's actual experience level

Generate the interview questions now:"""

# ============================================
# PROMPTS - ANSWER SCORING
# ============================================

ANSWER_EVALUATION_PROMPT = """You are evaluating a candidate's interview answer.

**QUESTION #{question_id} ({category}):**
{question}

**EXPECTED TOPICS:**
{expected_topics}

**CANDIDATE'S ANSWER:**
{answer}

**EVALUATION CRITERIA:**
1. Relevance - Does the answer address the question?
2. Completeness - Are key points covered?
3. Clarity - Is the explanation clear and well-structured?
4. Technical accuracy - Is the information correct?
5. Depth - Does it show good understanding?

**SCORING (0-10):**
- 9-10: Excellent, comprehensive, accurate
- 7-8: Good understanding, minor gaps
- 5-6: Basic understanding, missing key points
- 3-4: Partial/unclear understanding
- 0-2: No answer or completely wrong

**OUTPUT JSON:**
{{
  "score": 7.5,
  "feedback": "Detailed evaluation...",
  "strengths": ["specific strength 1", "specific strength 2"],
  "weaknesses": ["specific weakness 1"],
  "key_points_covered": ["point1", "point2"],
  "key_points_missed": ["point1"]
}}

Evaluate the answer:"""

OVERALL_SUMMARY_PROMPT = """Provide an overall assessment of the interview performance.

**CANDIDATE:** {candidate_name}
**OVERALL SCORE:** {overall_score}/100

**QUESTION-WISE PERFORMANCE:**
{performance_breakdown}

**SUMMARY GUIDELINES:**
- overall_feedback: 2-3 sentences on overall performance
- recommendation: strong-hire (80-100) / hire (65-79) / conditional-hire (50-64) / no-hire (<50)
- strengths_summary: Top 3 strengths shown
- areas_for_improvement: Top 3 areas needing work

**OUTPUT JSON:**
{{
  "overall_feedback": "Overall performance summary",
  "recommendation": "hire/no-hire/conditional-hire/strong-hire",
  "strengths_summary": ["strength 1", "strength 2", "strength 3"],
  "areas_for_improvement": ["area 1", "area 2", "area 3"]
}}

Generate summary:"""

# ============================================
# UTILITY FUNCTIONS
# ============================================

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return "".join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    try:
        docx_file = BytesIO(file_content)
        doc = Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting DOCX: {str(e)}")

def safe_json_parse(response_content: str) -> dict:
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        raise ValueError("No valid JSON found")

def create_resume_summary(resume_data: dict) -> str:
    """Create detailed resume summary for question generation"""
    parts = []
    
    # Personal info
    personal = resume_data.get("personal_info", {})
    name = personal.get("name", "Candidate")
    parts.append(f"CANDIDATE: {name}\n")
    
    # Education
    education = resume_data.get("education", [])
    if education:
        parts.append("EDUCATION:")
        for edu in education:
            degree = edu.get("degree", "")
            institution = edu.get("institution", "")
            field = edu.get("field_of_study", "")
            if degree or institution:
                edu_line = f"  • {degree}"
                if field:
                    edu_line += f" in {field}"
                if institution:
                    edu_line += f" from {institution}"
                parts.append(edu_line)
    
    # Experience
    experience = resume_data.get("experience", [])
    if experience:
        parts.append("\nPROFESSIONAL EXPERIENCE:")
        for exp in experience:
            role = exp.get("role", "")
            company = exp.get("company", "")
            parts.append(f"  • {role} at {company}")
            
            responsibilities = exp.get("responsibilities", [])
            if responsibilities:
                parts.append(f"    Responsibilities: {', '.join(responsibilities[:3])}")
            
            techs = exp.get("technologies_used", [])
            if techs:
                parts.append(f"    Technologies: {', '.join(techs)}")
            
            achievements = exp.get("key_achievements", [])
            if achievements:
                parts.append(f"    Achievement: {achievements[0]}")
    
    # Projects
    projects = resume_data.get("projects", [])
    if projects:
        parts.append("\nPROJECTS:")
        for proj in projects:
            title = proj.get("title", "Project")
            description = proj.get("description", "")
            parts.append(f"  • {title}")
            if description:
                parts.append(f"    Description: {description[:150]}")
            
            techs = proj.get("technologies", [])
            if techs:
                parts.append(f"    Tech Stack: {', '.join(techs)}")
            
            challenges = proj.get("challenges_solved", [])
            if challenges:
                parts.append(f"    Challenge Solved: {challenges[0]}")
    
    # Skills
    skills = resume_data.get("skills", {})
    if skills:
        parts.append("\nTECHNICAL SKILLS:")
        for category, items in skills.items():
            if items and isinstance(items, list):
                parts.append(f"  {category.replace('_', ' ').title()}: {', '.join(items)}")
    
    # Achievements
    achievements = resume_data.get("achievements", [])
    if achievements:
        parts.append(f"\nACHIEVEMENTS: {', '.join(achievements[:3])}")
    
    # Certifications
    certifications = resume_data.get("certifications", [])
    if certifications:
        parts.append("\nCERTIFICATIONS:")
        for cert in certifications:
            name = cert.get("name", "")
            issuer = cert.get("issuer", "")
            if name:
                parts.append(f"  • {name}" + (f" by {issuer}" if issuer else ""))
    
    return "\n".join(parts)

def validate_resume_based_questions(questions: List[dict], resume_summary: str) -> List[dict]:
    """
    Validate that questions reference actual resume content.
    Log warnings for potentially generic questions.
    """
    resume_lower = resume_summary.lower()
    
    # Common generic technology questions that shouldn't appear unless tech is in resume
    generic_tech_patterns = [
        (r'explain\s+(react|angular|vue|node)', ['react', 'angular', 'vue', 'node']),
        (r'what\s+is\s+(react|angular|vue|mongodb|sql)', ['react', 'angular', 'vue', 'mongodb', 'sql']),
        (r'tell\s+me\s+about\s+(react|python|java|javascript)', ['react', 'python', 'java', 'javascript']),
        (r'define\s+(oop|rest|api|database)', ['oop', 'rest', 'api', 'database'])
    ]
    
    for i, q in enumerate(questions[1:], start=2):  # Skip intro question
        question_lower = q['question'].lower()
        
        # Check for generic patterns
        for pattern, techs in generic_tech_patterns:
            if re.search(pattern, question_lower):
                # Check if any of these technologies are actually in resume
                tech_in_resume = any(tech in resume_lower for tech in techs)
                
                if not tech_in_resume:
                    logger.warning(
                        f"⚠️ Question {i} might be generic (not resume-based): {q['question']}"
                    )
                    logger.warning(f"   Context: {q.get('context', 'No context provided')}")
    
    return questions

def is_answer_valid(answer: str) -> bool:
    """Check if answer is meaningful"""
    if not answer or len(answer.strip()) < 10:
        return False
    words = answer.strip().split()
    if len(words) < 3:
        return False
    alpha_count = sum(c.isalpha() for c in answer)
    return alpha_count >= len(answer) * 0.6

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "message": "Resume-Based Natural Interview System v3.0",
        "endpoints": {
            "parse_resume": "/parse-resume",
            "generate_questions": "/generate-questions",
            "score_answers": "/score-answers",
            "health": "/health"
        }
    }

@app.post("/parse-resume", response_model=EnhancedResumeOutput)
async def parse_resume(file: UploadFile = File(...)):
    """Parse resume with consistent JSON structure"""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    allowed_types = ["application/pdf", "text/plain", 
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and DOCX files supported")
    
    try:
        file_content = await file.read()
        
        if file.content_type == "application/pdf":
            resume_text = extract_text_from_pdf(file_content)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(file_content)
        else:
            resume_text = file_content.decode('utf-8')
        
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        logger.info("Parsing resume...")
        prompt = RESUME_PARSING_PROMPT.format(resume_text=resume_text)
        
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        response = llm.invoke(prompt)
        resume_data = safe_json_parse(response.content)
        
        return EnhancedResumeOutput(**resume_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/generate-questions", response_model=InterviewQuestionSet)
async def generate_questions(resume_data: EnhancedResumeOutput):
    """Generate natural interview questions based purely on resume"""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    try:
        candidate_name = resume_data.personal_info.name or "the candidate"
        
        # Create detailed resume summary
        resume_summary = create_resume_summary(resume_data.model_dump())
        
        logger.info(f"Generating resume-based questions for {candidate_name}")
        logger.info(f"Resume summary length: {len(resume_summary)} chars")
        
        # Generate questions
        prompt = RESUME_BASED_QUESTION_GENERATION.format(
            resume_summary=resume_summary
        )
        
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=0.7,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        response = llm.invoke(prompt)
        questions_data = safe_json_parse(response.content)
        
        questions = questions_data.get("questions", [])
        
        # Validate questions are resume-based
        questions = validate_resume_based_questions(questions, resume_summary)
        
        # Validate and fix each question
        for i, q in enumerate(questions):
            if "expected_topics" not in q or not q["expected_topics"]:
                q["expected_topics"] = ["general"]
            if "category" not in q:
                q["category"] = "general"
            if "context" not in q:
                q["context"] = "Resume-based question"
        
        # Ensure first question is introduction
        if questions:
            questions[0] = {
                "id": 1,
                "question": "Tell me about yourself — your background, education, any experience you have, and projects you've worked on. Feel free to share any interests or hobbies as well.",
                "category": "introduction",
                "expected_topics": ["background", "education", "experience", "projects", "interests"],
                "context": "Opening introduction"
            }
        
        # Ensure we have at least 7 questions
        if len(questions) < 7:
            logger.warning(f"Only {len(questions)} questions generated, expected 7")
        
        logger.info(f"✅ Successfully generated {len(questions[:7])} questions")
        
        return InterviewQuestionSet(
            candidate_name=candidate_name,
            questions=questions[:7],  # Take first 7
            total_questions=len(questions[:7])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/score-answers", response_model=ScoringOutput)
async def score_answers(scoring_input: ScoringInput):
    """Score candidate's interview answers with detailed feedback"""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    if not scoring_input.interview_data:
        raise HTTPException(status_code=400, detail="No interview data provided")
    
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        question_scores = []
        questions_attempted = 0
        
        # Evaluate each question
        for qa in scoring_input.interview_data:
            # Check if answer is valid
            if not is_answer_valid(qa.answer):
                question_score = QuestionScore(
                    question_id=qa.question_id,
                    question=qa.question,
                    category=qa.category,
                    candidate_answer=qa.answer,
                    time_taken=qa.time_taken,
                    score=0.0,
                    feedback="No meaningful answer provided.",
                    strengths=[],
                    weaknesses=["No answer provided"],
                    key_points_covered=[],
                    key_points_missed=qa.expected_topics
                )
                question_scores.append(question_score)
                continue
            
            questions_attempted += 1
            
            # Generate evaluation prompt
            prompt = ANSWER_EVALUATION_PROMPT.format(
                question_id=qa.question_id,
                category=qa.category,
                question=qa.question,
                expected_topics=", ".join(qa.expected_topics),
                answer=qa.answer
            )
            
            # Get evaluation
            response = llm.invoke(prompt)
            eval_data = safe_json_parse(response.content)
            
            if not eval_data:
                eval_data = {
                    "score": 5.0,
                    "feedback": "Could not parse evaluation.",
                    "strengths": [],
                    "weaknesses": ["Evaluation failed"],
                    "key_points_covered": [],
                    "key_points_missed": qa.expected_topics
                }
            
            score = float(eval_data.get("score", 5.0))
            
            question_score = QuestionScore(
                question_id=qa.question_id,
                question=qa.question,
                category=qa.category,
                candidate_answer=qa.answer,
                time_taken=qa.time_taken,
                score=round(score, 2),
                feedback=eval_data.get("feedback", ""),
                strengths=eval_data.get("strengths", []),
                weaknesses=eval_data.get("weaknesses", []),
                key_points_covered=eval_data.get("key_points_covered", []),
                key_points_missed=eval_data.get("key_points_missed", [])
            )
            question_scores.append(question_score)
        
        # Calculate final score
        all_scores = [s.score for s in question_scores]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        overall_score = round((avg_score / 10) * 100, 2)
        
        final_score = FinalScore(overall_score=overall_score)
        
        # Generate overall summary
        performance_breakdown = "\n".join([
            f"Q{s.question_id} ({s.category}): {s.score}/10"
            for s in question_scores
        ])
        
        summary_prompt = OVERALL_SUMMARY_PROMPT.format(
            candidate_name=scoring_input.candidate_info.name,
            overall_score=overall_score,
            performance_breakdown=performance_breakdown
        )
        
        summary_response = llm.invoke(summary_prompt)
        summary_data = safe_json_parse(summary_response.content)
        
        if not summary_data:
            summary_data = {
                "overall_feedback": f"Candidate scored {overall_score}/100 overall.",
                "recommendation": "conditional-hire" if overall_score >= 50 else "no-hire",
                "strengths_summary": ["Requires review"],
                "areas_for_improvement": ["Requires review"]
            }
        
        logger.info(f"✅ Scoring complete: {overall_score}/100")
        
        return ScoringOutput(
            candidate_name=scoring_input.candidate_info.name,
            total_questions=len(scoring_input.interview_data),
            questions_attempted=questions_attempted,
            question_scores=question_scores,
            final_score=final_score,
            overall_feedback=summary_data.get("overall_feedback", ""),
            recommendation=summary_data.get("recommendation", "conditional-hire"),
            strengths_summary=summary_data.get("strengths_summary", []),
            areas_for_improvement=summary_data.get("areas_for_improvement", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scoring answers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "groq_configured": bool(GROQ_API_KEY),
        "features": [
            "Resume parsing with null for missing fields",
            "Natural resume-based questions (no difficulty levels)",
            "Answer scoring with detailed feedback",
            "No technology bias - purely resume-driven",
            "Question validation to ensure resume-based content"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)