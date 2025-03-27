import requests
from bs4 import BeautifulSoup
import logging
# Intern → 1
# Entry level (Junior) → 2
# Associate → 3
# Mid-Senior level → 4
# Director → 5
# Executive → 6

logger = logging.getLogger("interview")

class Scrape:
    """
    A web scraper for extracting job listings from LinkedIn."
    """
    def __init__(self,job:str,pos:list[int]):
        """
        Initializes the Scrape class with a job title and experience level filters.

        Args:
            job (str): The job title to search for.
            pos (list[int]): A list of experience level codes.
        """
        self.job=job
        self.pos=pos
        self.url = f"https://www.linkedin.com/jobs/search?keywords={self.job}&f_E={','.join(map(str,self.pos))}"
        self.space=[]
        
    def list_items(self) -> list:
        """
        Scrapes job listings from LinkedIn and extracts basic job elements.

        Returns:
            list: A list of extracted job elements (HTML divs).
        """
        
        basics=[]
        headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
        response=requests.get(self.url,headers)

        if response.status_code != 200:
            logger.error(f"Failed to fetch jobs. Status Code: {response.status_code}")
            return basics
        soup=BeautifulSoup(response.text,"html5lib")
        clean_content=soup.find_all("li")
        for each in clean_content:
            basics.append(each.find("div",{"class":"base-card"}))
        logger.info("Job scraping successful")
        return basics
    
        
    def data_extraction(self,n: int=10) -> list[dict]:

        """
        Extracts job details from LinkedIn job listings.

        Args:
            n (int): The maximum number of job listings to extract.

        Returns:
            list[dict]: A list of dictionaries containing job details.
        """
 
        for count,item in enumerate(self.list_items()):
            if item is None:
                continue
            try:
                store={"job":item.find("div",{"class":"base-search-card__info"})
                            .find("h3",{"class":"base-search-card__title"})
                            .get_text(strip=True) if item.find("div",{"class":"base-search-card__info"}) else "N/A",

                    "company":item.find("a",{"class":"hidden-nested-link"})
                                .get_text(strip=True) if item.find("a",{"class":"hidden-nested-link"}) else "N/A"
                        ,
                        "link":item.find("a",{"class":"base-card__full-link"})
                            .get("href") if item.find("a",{"class":"base-card__full-link"}) else "N/A",

                        "location":item.find("span",{"class":"job-search-card__location"})
                                .get_text(strip=True) if item.find("span",{"class":"job-search-card__location"}) else "N/A",
                        }
            except AttributeError as e:
                logger.warning(f"Skipping item {count} due to AttributeError: {e}")
                continue

            self.space.append(store)
            if count==n:
                break
        return self.space

