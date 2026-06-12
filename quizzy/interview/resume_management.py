from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from langchain_ollama import OllamaEmbeddings
import numpy as np
from pydantic import BaseModel, Field
import logging
from . import llm

logger = logging.getLogger("interview")

class CandidateInfo(BaseModel):
    """Structured response for candidate information extraction."""
    candidate: str = Field(..., description="Candidate's name")
    job: str = Field(..., description="Job title")
    skills: list[str] = Field(..., description="List of key technical and soft skills")
    experience: str = Field(..., description="Brief overview of professional experience")
    summary: str = Field(..., description="A 2-3 sentence professional summary of the candidate's profile")

class Resume:
    """
    A class for processing resumes, extracting information, and calculating ATS similarity scores.
    """    
    def resume_reader(self,file_name: object) -> str:
        """
        Reads and extracts text from a given PDF file.

        Args:
            file_name (object): The PDF file object.

        Returns:
            str: Extracted text from the PDF.
            None: If an error occurs during extraction.
        """

        output_string = StringIO()

        try:  
            parser = PDFParser(file_name)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
            out=output_string.getvalue()
            self.resume=out
            logger.info("PDF extraction successful")
            return out
        except Exception as e:
            logger.error(f"PDF text extraction failed : {str(e)}")
            return None

    async def domain_name_extraction(self,text: str) -> dict:
        """
        Extracts candidate's details from the given text using an LLM model.

        Args:
            text (str): The input text containing the resume and job description.

        Returns:
            dict: A dictionary with the extracted candidate name, job title, skills, experience, and summary.
            Defaults to placeholder values if extraction fails.
        """ 
        try:
            system_prompt = "You are a data extraction assistant. Extract the candidate's name, job title, skills, experience overview, and a brief professional summary from the provided text."
            user_content = text
            
            response = await llm.client.beta.chat.completions.parse(
                model=llm.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=CandidateInfo,
                temperature=llm.TEMPERATURE
            )
            parsed_output = response.choices[0].message.parsed
            
            # Post-process to guarantee only the first name is returned
            first_name = parsed_output.candidate.split()[0] if parsed_output.candidate else "candidate"
            
            result_dict = parsed_output.model_dump()
            result_dict["candidate"] = first_name

            logger.info("Candidate detail extraction successful")
            return result_dict

        except Exception as e:
            logger.error(f"Candidate detail extraction failed :{str(e)}")
            return {
                    "candidate":"candidate",
                    "job":"job",
                    "skills": [],
                    "experience": "Not available",
                    "summary": "Not available"
                }

    def ats_score_checker(self,file: str,des: str) -> int:
        """
        Computes an ATS (Applicant Tracking System) score using cosine similarity 
        between the resume text and job description.

        Args:
            file (str): The extracted resume text.
            des (str): The job description text.

        Returns:
            int: The ATS score as a percentage (0-100).
        """
        try:    
            embeddings = OllamaEmbeddings(model="twine/mxbai-embed-xsmall-v1")
            vector1 = np.array(embeddings.embed_query(file))
            vector2 = np.array(embeddings.embed_query(des))
            score = np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
            logger.info("ATS calculation successful")
            return round(score*100,2)
        except Exception as e:
            logger.error(f"ATS score calculation failed :{str(e)}")
            return 0
           

    # def summarize_resume(self, text_to_summarize: str) -> str:
    #     """
    #     Summarizes the resume text using a pre-trained transformer model.
    #     Note: This is currently commented out to save processing power.
    #     """
    #     from transformers import pipeline
    #     
    #     # Load a default pre-trained summarization pipeline (e.g., T5-small)
    #     summarizer = pipeline("summarization", model="t5-small")
    #     
    #     # Generate summary
    #     summary = summarizer(text_to_summarize, max_length=150, min_length=40, do_sample=False)
    #     
    #     # Extract and return the summary text
    #     summary_text = summary[0]['summary_text']
    #     print(summary_text)
    #     return summary_text

    async def final(self,file: object, description: str) -> tuple | None:
        """
        Processes the resume file and job description, returning the ATS score, 
        extracted resume text, job description, and extracted candidate details.

        Args:
            file (object): The resume file object.
            description (str): The job description text.

        Returns:
            tuple: (ATS score, resume text, job description, extracted candidate details)
            None: If resume extraction fails.
        """
        resume=self.resume_reader(file)
        if resume:
            text="\n".join([resume[:200].ljust(200),description[:500].ljust(500)])
            dictionary = await self.domain_name_extraction(text)
            score=self.ats_score_checker(resume,description)
            return score, resume, description, dictionary
        else:
            return None