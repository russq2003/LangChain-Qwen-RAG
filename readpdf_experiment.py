import pdfplumber


def load_pdf(path):
    text = ''
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + '\n'
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    text += ", ".join(row) + "\n"
    return text

path = 'D:\EEagent1\doc\大创结题报告-final.pdf'
text = load_pdf(path)

print(text)
