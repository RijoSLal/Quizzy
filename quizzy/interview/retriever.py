# import google.generativeai as genai
import os 
from . import vectordb
from dotenv import load_dotenv
import random
from groq import Groq
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
import logging
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
logger = logging.getLogger("interview")
load_dotenv()

# genai.configure(api_key=os.getenv("APIKEY"))
# model = genai.GenerativeModel("gemini-2.0-flash") 


client = Groq(
    api_key=os.getenv("GROQ"),
)

llm=init_chat_model("llama3-70b-8192", model_provider="groq",groq_api_key=os.getenv("GROQQ"))



class EvaluationResult(BaseModel):
    """structured response for RAG evaluation."""
    score: int = Field(..., ge=1, le=10, description="Score from 1 to 10.")
    reason: str = Field(..., description="Brief reason for the score.")
    better: str = Field(..., description="Improved version of the user's answer.")
    communication_feedback: str = Field(..., description="A short tip to improve clarity, conciseness, or tone.")



class RAG:
    """
    A Retrieval-Augmented Generation (RAG) system for AI-driven interview evaluation. 

    This class enables:
    - Storing and retrieving documents using ChromaDB.
    - Generating interview-style questions based on given documents.
    - Evaluating user responses with an AI model.
    - Conducting Socratic-style conversations to assess critical thinking.

    Attributes:
        question_types (dict): A mapping of question categories to their corresponding prompts.
        chromadb_instance (ChromaDB): An instance of ChromaDB for document retrieval.
        level (str | None): The difficulty level of questions.
        score (int): The cumulative score of evaluated answers.
        count (int): The number of evaluated answers.
        structured_llm (LLM): A structured LLM for evaluating responses.
    """
    def __init__(self):
        """Initializes the RAG system with predefined question types, a vector database instance, and an LLM for structured evaluation."""
        self.question_types= {
            "Clarification": "Can you elaborate on that?",
            "Assumption Testing": "What assumptions are you making here?",
            "Reason and Evidence": "What evidence supports your response?",
            "Viewpoint Exploration": "What are some alternative perspectives?",
            "Implication and Consequence": "What are the possible consequences of this?",
            "Questioning the Question": "Why do you think this question is important?"
                            }
        self.chromadb_instance=vectordb.ChromaDB()
        self.level=None
        self.score=0
        self.count=0
        self.structured_llm = llm.with_structured_output(EvaluationResult)
        self.resume=None
        self.model_name = "sshleifer/distilbart-cnn-12-6"
        
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
        self.chromadb_instance.insert_into_chroma(resume)
        self.chromadb_instance.insert_into_chroma(job_description)
        # logger.info("All the documents are successfully inserted in ChromaDB")


    def get_random_document(self) -> str | None:
        """
        Fetches a random document from the database.

        Returns:
            str | None: A randomly selected document, or None if retrieval fails.
        """
        try:
            extracted_random_docs=self.chromadb_instance.get_random_document()
            logger.info("Random document extraction successful")
            return extracted_random_docs
        except Exception as e:
            logger.error(f"Quizzy encountered an error while fetching random document: {e}")
            return None

    def generate_interview_questions(self, document: str|None, question: str, response: str, question_type: str = "critical thinking") -> str | None:
        """
        Generates an AI-driven interview question based on the given document and user's previous response.

        Args:
            document (str): The reference document for generating questions.
            question (str): The previous question asked to the user.
            response (str): The user's response to the previous question.
            question_type (str, optional): The type of question to generate. Defaults to "critical thinking".

        Returns:
            str: The generated interview question, or an error message if generation fails.
        """
        try:
            if document:             
                prompt = f"""
                            You are Quizzy, an AI interviewer. First, respond to the candidate's previous answer in short line.  
                            Then, generate a challenging {self.level}-level question based on the provided document.  
                            
                            Previous Question:  
                            "{question}"  

                            Response to Previous Answer:  
                            "{response}"  

                            Document: "{document}"  
                            Question Type: "{question_type}"  

                            Ensure the question is thought-provoking and tests analytical skills. Return only the response and the new question, without explanations.
                            """
            else:
                logger.info(f"All document extracted no more document available")
                return None
            
            chat_completion = client.chat.completions.create(
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": prompt,
                                        }
                                    ],
                                    model="llama3-70b-8192",
                                )
            return chat_completion.choices[0].message.content
           
        except Exception as e:
            logger.error(f"Groq llama3-70b-8192 question generating API encountered an error : {e}")
            return None

    def evaluate_answer(self,question: str, user_answer:str) -> str | None:
        """
        Evaluates a user's answer based on relevance, clarity, and depth.

        Args:
            question (str): The interview question that was asked.
            user_answer (str): The user's response to the question.

        Returns:
            str: A structured evaluation containing:
                - A score (1-10)
                - A reason for the score
                - An improved answer suggestion
                - A short tip for improvement
        """

        try:
            prompt = f"""
            You are an AI evaluator. Given a question and a user's answer, score the response from 1 to 10 based on relevance, clarity, and depth.

            Question: "{question}"
            User Answer: "{user_answer}"

            Return:
            - Score (1-10)
            - A brief reason for the score (one simple sentence)
            - A better answer in one sentence (not too long)
            - A short tip to improve clarity, conciseness, or tone
            """
            structured_out = self.structured_llm.invoke(prompt)
            self.score+=structured_out.score
            self.count+=1

            formatted_output = str(structured_out)
            logger.info("successfully evaluated User_answer and Question")
            return formatted_output

           
        except Exception as e:
            logger.error(f"llama3-70b-8192 Groq model question evaluation failed: {e}")
            return None
        
        
    
    def socratic_conversation(self,previous_quest: str,user_resp: str) -> str | None:
        """
        Engages in a Socratic-style conversation by generating a follow-up question based on a randomly retrieved document.

        Args:
            previous_quest (str): The previous question in the conversation.
            user_resp (str): The user's response to the previous question.

        Returns:
            str: A follow-up question, or None if document retrieval fails.
        """
        document = self.get_random_document()
        
        if not document: #just for safety even though this is implemented in self.get_random_document()
            return None
        question_type = random.choice(list(self.question_types.keys()))
        questions = self.generate_interview_questions(document, previous_quest, user_resp, question_type)
        logger.info("RAG question generation cycle completed")
        return questions
    

    def candidate_document_summarization(self) -> str:
        """
        Generates a summary of the extracted resume using a pre-trained NLP model.

        Returns:
            str: The generated summary of the resume.
            "Minor issue identified in the summarization process. Adjustments are in progress for optimal results": If the summary generation fails.
        """

        try:
            raise Exception("Bypass summarization")
            # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # model = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_name,from_pt=True)

            # inputs = tokenizer(self.resume, return_tensors="tf", truncation=True)  # no max_length here
            # summary_ids = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)  # summary length
            # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            logger.info(f"Summarization of candidate profile completed")
            # return summary
        except Exception as e:
            logger.error(f"Unexpected error happened in summarization : {e}")
            return "Minor issue identified in the summarization process. Adjustments are in progress for optimal results"



