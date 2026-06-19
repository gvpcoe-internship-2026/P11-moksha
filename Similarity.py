from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def read_jd(jd_path):
    with open(jd_path, "r", encoding="utf-8") as file:
        return file.read()


def calculate_similarity(model, resume_text, jd_text):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    similarity = cos_sim(resume_embedding, jd_embedding)
    return similarity.item()


model = SentenceTransformer("all-MiniLM-L6-v2")

resume_text = extract_text_from_pdf("samples/my_resume.pdf")
python_jd = read_jd("samples/jd_python.txt")
civil_jd = read_jd("samples/jd_civil.txt")

python_similarity = calculate_similarity(model, resume_text, python_jd)
civil_similarity = calculate_similarity(model, resume_text, civil_jd)

print(f"Resume vs Python Developer JD: {python_similarity:.4f}")
print(f"Resume vs Civil Engineer JD: {civil_similarity:.4f}")