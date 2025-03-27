from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
import ollama
import numpy as np
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_ollama import OllamaLLM
import logging

logger = logging.getLogger("interview")

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

    def domain_name_extraction(self,text: str) -> dict:
        """
        Extracts the candidate's name and job title from the given text using an LLM model.

        Args:
            text (str): The input text containing the resume and job description.

        Returns:
            dict: A dictionary with the extracted candidate name and job title.
            Defaults to {'candidate': 'candidate', 'job': 'job'} if extraction fails.
        """ 
        try:
            llm = OllamaLLM(model="llama3.2:1b")

            schemas = [
                ResponseSchema(name="candidate", description="Candidate's name"),
                ResponseSchema(name="job", description="Job title"),
            ]

            parser = StructuredOutputParser.from_response_schemas(schemas)

            text += f"\n{parser.get_format_instructions()}"

            # get and parse response
            parsed_output = parser.parse(llm.invoke(text))
            logger.info("Domain&Name extraction successful")
            return parsed_output

        except Exception as e:
            logger.error(f"Domain&Name extraction failed :{str(e)}")
            return {"candidate":"candidate",
                    "job":"job"}

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
            vector1 = np.array(ollama.embeddings("mxbai-embed-large",file)['embedding'])
            vector2 = np.array(ollama.embeddings("mxbai-embed-large", des)['embedding'])
            score=np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
            logger.info("ATS calculation successful")
            return round(score*100,2)
        except Exception as e:
            logger.error(f"ATS score calculation failed :{str(e)}")
            return 0
           

    def final(self,file: object, description: str) -> tuple | None:
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
            dictionary=self.domain_name_extraction(text)
            score=self.ats_score_checker(resume,description)
            return score, resume, description, dictionary
        else:
            return None
        

   

    
    

  

