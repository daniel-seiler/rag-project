import re
import os
import pandas as pd
from enum import Enum
from bs4 import BeautifulSoup


DOC_LOCATION = os.environ["PCL_PATH"]
DOC_STARTPOINT = "modules.html"

df = pd.DataFrame(columns=['name', 'depth', 'type', 'parent', 'source', 'description'])


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
    href_pattern = re.compile("https?://")
    for a in soup.find_all("a"):
        if not href_pattern.match(a['href']):
            links.append(a['href'])
    return links


def analyse_code(soup: BeautifulSoup, line: str, name: str, parent: str, source: str) -> list:
    code = soup.find("a", {"name": line}).parent
    code_block_started = False
    open_parenthesis = 0
    code_content = ''
    while (not code_block_started) or (open_parenthesis != 0):
        if code is None:
            break
        code.find_next("span", {"class": "lineno"}).decompose()
        new_line = code.get_text(strip=True)
        code_content += new_line
        code_content += "\n"
        if "{" in new_line:
            code_block_started = True
            open_parenthesis += 1
        if "}" in new_line:
            open_parenthesis -= 1
        code = code.find_next("div", {"class": "line"})
    return [
        name,
        2,
        DocTypes.CODE.value,
        parent,
        source,
        code_content
    ]


def analyse_description(soup: BeautifulSoup, depth: int, parent: str, source: str) -> list:
    name = soup.get_text()[2:]
    data_type = DocTypes.get_type_from_header(soup.find_previous("h2", {"class": "groupheader"}).get_text())
    data_description = soup.find_next("div", {"class": "memdoc"})
    data = [[
        name,
        depth,
        data_type.value,
        parent,
        source,
        data_description.get_text()
    ]]
    code = data_description.find("a", {"class": "el"})
    href_pattern = re.compile("(?P<path>.*)#(?P<line>l\d+)")
    if data_type is DocTypes.FUNCTION and code is not None and href_pattern.match(code["href"]):
        match = href_pattern.match(code["href"])
        with open(DOC_LOCATION + match.group('path'), "r") as f_:
            code_soup = BeautifulSoup(f_, "html.parser")
        data.append(analyse_code(code_soup, match.group('line'), name, parent, code["href"]))
    return data


def analyse_class(soup: BeautifulSoup, depth: int, doc_type: DocTypes, parent: str, source: str):
    global df
    description_list = []
    for element in soup.find("div", {"class": "contents"}):
        if element.name == "div" and "memitem" in element.get("class", []):
            break
        description_list.append(element.get_text())
    description = ''.join(description_list).strip()
    df.loc[len(df)] = [
        soup.find("div", {"class": "title"}).get_text(),
        depth,
        doc_type.value,
        parent,
        source,
        description
    ]
    # Analyse contents
    for detailed_description in soup.find_all("h2", {"class": "memtitle"}):
        df = pd.concat([pd.DataFrame(analyse_description(detailed_description,
                                                         depth + 1,
                                                         parent,
                                                         source),
                                     columns=df.columns),
                         df])


# Get all modules:
with open(DOC_LOCATION + DOC_STARTPOINT, "r") as f:
    module_locations = get_internal_links(BeautifulSoup(f, "html.parser"))

# Analyse module page
for module_location in module_locations:
    with open(DOC_LOCATION + module_location, "r") as f:
        module_soup = BeautifulSoup(f, "html.parser")
    analyse_class(module_soup, 0, DocTypes.MODULE, module_location, module_location)

    # Find all paths leading to other elements
    further_paths = set()
    for table_element in module_soup.find_all("table", {"class": "memberdecls"}):
        for path in table_element.find_all("a", {"class": "el"}):
            if module_location not in path['href'] and "#" not in path['href']:
                further_paths.add(path['href'])

    # Analyse the found elements
    for path in further_paths:
        with open(DOC_LOCATION + path, "r") as f:
            page = BeautifulSoup(f, "html.parser")
        analyse_class(page, 1, DocTypes.CLASS, module_location, path)

# Write data to csv
df.to_csv("data.csv", index=False)
