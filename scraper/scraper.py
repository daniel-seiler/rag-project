from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from enum import Enum

DOC_LOCATION = "/home/uncreative/Git/pcl-documentation/"    #TODO remove absolute path
DOC_STARTPOINT = "modules.html"

df = pd.DataFrame(columns=['name', 'type', 'parent', 'source', 'description'])

class DocTypes(Enum):
    MODULE = "module"
    CLASS = "class"
    STRUCT = "struct"
    MACRO = "macro"
    TYPEDEF = "typedef"
    ENUM = "enum"
    CONSTRUCTOR = "constructor"
    FUNCTION = "function"
    MEMBER = "member"
    CODE = "code"

    @classmethod
    def get_type_from_header(cls, header: str):
        for t in DocTypes:
            if t.value in header.lower():
                return t
        print("Error: type for {} not found".format(header))
        return None

def get_internal_links(soup: BeautifulSoup) -> list:
    links = []
    href_pattern = re.compile("https?://")  #TODO make regex as parameter
    for a in soup.find_all("a"):
        if not href_pattern.match(a['href']):
            links.append(a['href'])
    return links

def get_code(soup: BeautifulSoup, name: str, parent: str, source: str) -> list:
    return [
        name,
        DocTypes.CODE.value,
        parent,
        source,
    ]

def analyse_description(soup: BeautifulSoup, parent: str, source: str) -> list:
    name = soup.get_text()[2:]
    data_type = DocTypes.get_type_from_header(soup.find_previous("h2", {"class": "groupheader"}).get_text())
    description = soup.find_next("div", {"class": "memdoc"})
    data = [[
        name,
        data_type.value,
        parent,
        source,
        description.get_text()
    ]]
    code = description.find("a", {"class": "el"})
    href_pattern = re.compile(".*#l\d+")
    if data_type is DocTypes.FUNCTION and code is not None and href_pattern.match(code["href"]):
        data.append(get_code(soup, name, parent, source))
    return data


# Get all modules:
with open(DOC_LOCATION + DOC_STARTPOINT, "r") as f:
    modules = get_internal_links(BeautifulSoup(f, "html.parser"))

# Analyse module page
for module_location in modules:
    with open(DOC_LOCATION + module_location, "r") as f:
        module = BeautifulSoup(f, "html.parser")
    data_to_append = [
        module.find("div", {"class": "title"}).get_text(),
        DocTypes.MODULE.value,
        None,
        module_location
    ]
    description = module.find("h1")
    if description is not None:
        data_to_append.append(description.find_next("p").get_text())
    else:
        data_to_append.append(None)
    df.loc[len(df)] = data_to_append

    # Analyse detailed documentations
    for detailed_description in module.find_all("h2", {"class": "memtitle"}):
        df = pd.concat([pd.DataFrame(analyse_description(detailed_description,
                                                         module.find("div", {"class": "title"}).get_text(),
                                                         module_location),
                                     columns=df.columns),
                        df])

    # Analyse classes
    # Find links to classes
    further_links = set()
    for table in module.find_all("table", {"class": "memberdecls"}):
        for link in table.find_all("a", {"class": "el"}):
            if module_location not in link['href'] and "#" not in link['href']:
                further_links.add(link['href'])

    # Analyse single class
    for link in further_links:
        with open(DOC_LOCATION + link, "r") as f:               #TODO use this for module?
            page = BeautifulSoup(f, "html.parser")
        description_list = []
        for element in page.find("div", {"class": "contents"}):
            if element.name == "div" and "memitem" in element.get("class", []):
                break
            description_list.append(element.get_text())
        description = ''.join(description_list).strip()
        # Analyse contents
        for detailed_description in page.find_all("h2", {"class": "memtitle"}):
            df = pd.concat([pd.DataFrame(analyse_description(detailed_description,
                                                             page.find("div", {"class": "title"}).get_text(),
                                                             link),
                                         columns=df.columns),
                            df])
df.to_csv("data.csv", index=False)
