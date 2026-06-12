# import google.generativeai as genai
import os 
from . import vectordb, llm
from dotenv import load_dotenv
import random
from pydantic import BaseModel, Field
import logging
import orjson
from django.conf import settings
import time

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("interview")

class SocraticQuestion(BaseModel):
    """A structured representation of a technical question."""
    question: str = Field(..., description="The technical question text.")
    category: str = Field(..., description="The Socratic category of the question.")

class StructuredQuestions(BaseModel):
    """List of pre-generated interview questions."""
    questions: list[SocraticQuestion] = Field(..., description="A list of exactly 15 UNIQUE and challenging interview questions. No duplicates.")


class RAG:
    """
    A Retrieval-Augmented Generation (RAG) system for AI-driven interview evaluation. 

    This class enables:
    - Storing and retrieving documents using ChromaDB.
    - Preparing a structured list of interview questions.
    - Evaluating user responses with an AI model.
    - Conducting Socratic-style conversations using pre-generated questions.

    Attributes:
        chromadb_instance (ChromaDB): An instance of ChromaDB for document retrieval.
        level (str | None): The difficulty level of questions.
        score (int): The cumulative score of evaluated answers.
        count (int): The number of evaluated answers.
    """
    def __init__(self, session_id: str, level: str = None):
        """Initializes the RAG system with a vector database instance specific to the user session."""
        self.chromadb_instance = vectordb.ChromaDB(session_id)
        self.level=level
        self.score=0
        self.count=0
        self.resume=None
        self._load_fallback_questions()
        
    def _load_fallback_questions(self):
        """Loads fallback questions and goodbyes from an external JSON file using orjson."""
        try:
            # Construct path relative to the project root (assuming assets is in the root)
            assets_path = os.path.join(settings.BASE_DIR, 'assets', 'fallback_questions.json')
            with open(assets_path, 'rb') as f:
                raw_data = orjson.loads(f.read())
                
                # Handle both old and new JSON formats for robustness
                if isinstance(raw_data, list):
                    # Old format: direct list of questions
                    self.fallback_questions = [{"question": q["question"], "category": q["category"], "is_fallback": True} for q in raw_data]
                    self.fallback_goodbyes = ["Your interview is complete, let's see the result."]
                else:
                    # New format: dict with "questions" and "goodbyes" keys
                    self.fallback_questions = [{"question": q["question"], "category": q["category"], "is_fallback": True} for q in raw_data.get("questions", [])]
                    self.fallback_goodbyes = raw_data.get("goodbyes", ["Your interview is complete, let's see the result."])
                
            logger.info(f"Successfully loaded {len(self.fallback_questions)} fallback questions and {len(self.fallback_goodbyes)} goodbyes.")
        except Exception as e:
            logger.error(f"Failed to load fallback questions: {e}")
            self.fallback_questions = [
                {"question": "Explain the architectural trade-offs in your most recent project.", "category": "Architecture", "is_fallback": True}
            ]
            self.fallback_goodbyes = ["Your interview is complete, let's see the result."]

    def score_reset(self):
        """
        Reset score and count for the next iter
        """
        self.score=0
        self.count=0

        
    def document_insertion_chroma(self,resume: str,job_description: str) -> None:
        """
        Inserts the given resume and job description into ChromaDB after deleting any existing documents.

        Args:
            resume (str): The resume text to be inserted.
            job_description (str): The job description text to be inserted.

        Returns:
            None
        """
        self.score=0 #ensure score gets reset to zero every new document insertion
        logger.info("resetting score")
        self.resume = resume
        self.chromadb_instance.delete_inserted_docs()
        self.chromadb_instance.insert_into_chroma(resume, metadata={"source": "resume"})
        self.chromadb_instance.insert_into_chroma(job_description, metadata={"source": "job_description"})

    async def prepare_questions(self, resume: str, job_description: str) -> list[dict]:
        """
        Generates 15 meaningful technical interview questions.
        Includes a 30-second timeout and uses pre-loaded fallback questions.
        """
        import time
        import asyncio
        start_time = time.perf_counter()

        try:
            logger.info("Starting to prepare questions pool with 30s timeout...")
            system_prompt = (
                "You are a Senior Technical Interviewer. "
                "Generate a list of EXACTLY 15 UNIQUE, focused technical questions. "
                "Each question should focus on ONE specific concept from the candidate's domain or the job requirements. "
                "Avoid 'barrage' questions that ask about 5 different things at once. "
                "The questions should be deep but concise, testing practical implementation and architectural trade-offs. "
                f"Target difficulty: {self.level} level. "
                "Every question must be distinctly different from the others."
            )
            user_content = f"Resume: {resume}\nJob Description: {job_description}"

            # Wrap in timeout
            response = await asyncio.wait_for(
                llm.client.beta.chat.completions.parse(
                    model=llm.MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    response_format=StructuredQuestions,
                    temperature=llm.TEMPERATURE
                ),
                timeout=30.0
            )
            structured_out = response.choices[0].message.parsed
            
            # Length validation
            if len(structured_out.questions) < 12:
                logger.warning(f"SWITCHING TO MANUAL FALLBACK: LLM generated only {len(structured_out.questions)} questions (needed at least 12).")
                return self.fallback_questions

            duration = time.perf_counter() - start_time
            logger.info(f"Structured questions pool ({len(structured_out.questions)}) prepared successfully in {duration:.2f}s")
            return [{"question": q.question, "category": q.category, "is_fallback": False} for q in structured_out.questions]

        except asyncio.TimeoutError:
            logger.warning(f"SWITCHING TO MANUAL FALLBACK: LLM question generation timed out after {time.perf_counter() - start_time:.2f}s.")
            return self.fallback_questions
        except Exception as e:
            logger.error(f"SWITCHING TO MANUAL FALLBACK: Error preparing questions after {time.perf_counter() - start_time:.2f}s. Exception: {e}")
            return self.fallback_questions

    def get_system_prompt(self, is_final: bool, domain: str = None, level: str = None) -> str:
        """Returns the system prompt for the AI interviewer."""
        if is_final:
            return (
                "The interview is now over. "
                "You MUST say EXACTLY: 'Your interview is complete, we will give you the result shortly.' and nothing else."
            )
        
        # Map internal level names to seniority labels
        level_map = {
            "beginner": "SD1",
            "intermediate": "SD2",
            "advance": "SD3",
            "expert": "SD4/Expert"
        }
        seniority = level_map.get(level, level)
        
        persona = f"an AI {domain} technical interviewer" if domain else "a human-like AI technical interviewer"
        level_info = f" ({seniority} level)" if seniority else ""
        target = f"a candidate for a {domain} role{level_info}" if domain else f"a candidate{level_info}"

        return (
            f"You are Quizzy, {persona}. "
            f"You are currently interviewing {target}. "
            "Ask concise questions. "
            "Do NOT repeat the candidate's name or a 'Hi [Name]' greeting at the start of every message. "
            "Simply acknowledge the candidate's answer naturally (e.g., 'Okay', 'That makes sense', 'I see') and THEN ask the next technical question. "
            "If the candidate's response is completely unrelated to the interview (e.g., talking about the weather or personal life), politely steer them back. "
            "Otherwise, if they answer 'I don't know' or give a short answer, acknowledge it professionally and THEN ask the next technical question. "
            "CRITICAL: You MUST output the actual text of the next question. Do not just say 'Let's move to the next question'. You must actually ask it."
            "Do NOT end the interview."
        )

    async def socratic_conversation(self, chat_history: list[dict], suggested_questions: list[dict], candidate_name: str = None, domain: str = None, level: str = None) -> str | None:
        """
        Engages in a Socratic-style conversation using the full history and real-time RAG context.
        """
        try:
            # Prepare message list from history (avoiding shallow copy issues)
            messages = [dict(msg) for msg in chat_history]

            # Determine if this is the final turn
            is_final = not suggested_questions
            
            # IF FINAL TURN, DO NOT CALL LLM. Return a random predefined goodbye.
            if is_final:
                return random.choice(self.fallback_goodbyes)

            # Determine the query for VDB based on the next question or topic
            query_str = "technical interview"
            if suggested_questions:
                query_str = suggested_questions[0].get("question", query_str)

            # Retrieve relevant context from Chroma based on the question/topic
            search_results = self.chromadb_instance.query_vdb(query_str, k=1)
            
            if search_results:
                raw_context = search_results[0]
                source = raw_context.get("metadata", {}).get("source", "unknown")
                prefix = "[RESUME INFO]" if source == "resume" else "[JOB DESCRIPTION]" if source == "job_description" else "[CONTEXT]"
                reference_context = f"{prefix} {raw_context['content']}"
            else:
                reference_context = "N/A"
            
            system_prompt = self.get_system_prompt(is_final, domain=domain, level=level)
            
            # Ensure the latest system prompt is at the start
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = system_prompt
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            # Format instructions based on context
            if suggested_questions:
                current_q = suggested_questions[0]
                if current_q.get("is_fallback"):
                    pool_instruction = (
                        f"Topic: '{current_q['question']}' (Category: {current_q['category']}).\n"
                        "INSTRUCTION: You must ask the candidate ONE technical question about this specific topic. "
                        "Use the 'Reference Context' to make the question relevant. Do not output anything else."
                    )
                else:
                    pool_instruction = (
                        f"Target Question to ask: '{current_q['question']}'\n"
                        "INSTRUCTION: You MUST output this exact question now. You may add a brief 1-sentence acknowledgment of their previous answer before asking it, but YOU MUST ASK THE QUESTION."
                    )
            
            # Combine the final instruction with the last user message to maintain Assistant -> User turn consistency
            instruction_block = f"Reference Context: {reference_context}\n\nINSTRUCTION: {pool_instruction}"
            
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += f"\n\n{instruction_block}"
            else:
                messages.append({"role": "user", "content": instruction_block})
            
            logger.debug(f"Socratic Turn - Final: {is_final}")
            
            response = await llm.client.beta.chat.completions.create(
                model=llm.MODEL,
                messages=messages,
                temperature=llm.TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in socratic conversation: {e}")
            return None

    async def evaluate_answer(self, question: str, user_answer: str) -> dict | None:
        """
        Evaluates a candidate's answer based on the question and context.
        """
        try:
            system_prompt = (
                "You are an expert interviewer. Evaluate the candidate's answer based on the provided question. "
                "Return a JSON object with: 'score' (1-10), 'reason' (brief explanation), 'better' (a superior version of the answer), "
                "and 'communication_feedback' (feedback on how they expressed themselves)."
            )
            user_content = f"Question: {question}\nCandidate Answer: {user_answer}"
            
            class Evaluation(BaseModel):
                score: int
                reason: str
                better: str
                communication_feedback: str

            response = await llm.client.beta.chat.completions.parse(
                model=llm.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=Evaluation,
                temperature=llm.TEMPERATURE
            )
            structured_out = response.choices[0].message.parsed
            return {
                "score": structured_out.score,
                "reason": structured_out.reason,
                "better": structured_out.better,
                "communication_feedback": structured_out.communication_feedback
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
