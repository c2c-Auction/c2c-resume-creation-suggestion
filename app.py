import logging
from typing import Dict, Any, List, Tuple
from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
import os
from langchain_community.document_loaders import PyPDFLoader
import apis as a
import re
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompt import profile_generator_experience, profile_generator_prompt_no_experience

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

pdf_prompt = """
You have to find these things from the resume 

Basic Information
Full Name
Contact Information:
Phone Number
Email Address
Address (if available)
LinkedIn Profile (if available)
Personal Website or Portfolio (if available)
Professional Summary or Objective

Education
Degree(s) Obtained
Field(s) of Study
Institution(s) Attended
Graduation Date(s)
Academic Achievements or Honors

Work Experience
Job Title(s)
Company Name(s)
Location(s)
Employment Dates
Key Responsibilities and Achievements
Technologies and Tools Used

Skills
Technical Skills
Soft Skills
Certifications and Licenses

Projects
Project Title(s)
Description of Projects
Technologies and Tools Used
Role in the Project

Awards and Honors
Award Title(s)
Issuing Organization
Date of Award

Professional Development
Certifications
Training Programs
Workshops and Seminars

Publications and Research
Publication Titles
Journal or Conference Name
Date of Publication
Co-authors (if any)

Professional Affiliations
Membership in Professional Organizations
Leadership Roles in Professional Organizations

Languages
Languages Known
Proficiency Level

References
Reference Name(s)
Contact Information
Relationship to the Candidate

Additional Sections
Volunteer Experience
Hobbies and Interests
Personal Projects
Additional Information Relevant to the Job
"""

def ques(user_resume: str, job_description: str, work: str) -> str:
    return f"""
    You're an AI with a talent for crafting thoughtful prompts to extract necessary details from users for creating tailored resumes. Your task is to ask the user for three specific questions based on their provided resume, job description, and additional work information to generate a new resume.
    -----
        user_resume = {user_resume}
        ----
        job_description = {job_description}
        ----
        additional_work = {work}
        ----

    Please ask the user three questions that will help you create a customized and effective resume based on the information provided. Remember to consider the user's skills, experience, and the specific requirements of the job role outlined in the job description. This will ensure that the new resume you generate is a perfect fit for the user's desired position. questions should be sprated by | 
    example of format -> **question1** ** question2**  **question3 **
    """

def extract_between_asterisks(text: str) -> List[str]:
    pattern = r'\*\*(.*?)\*\*'
    return re.findall(pattern, text)

def input_pdf_setup(uploaded_file) -> Tuple[str, str]:
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(filepath)
    loader = PyPDFLoader(file_path=filepath)
    pages = loader.load_and_split()
    if len(pages) < 1:
        raise ValueError("The PDF file has no pages.")
    text = " ".join([page.page_content for page in pages])
    content = f"{pdf_prompt} here is the content of resume {text}"
    return a.final(content), filepath


def save_to_temp_file(data: Any) -> str:
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
        json.dump(data, temp_file, ensure_ascii=False)
        temp_file_path = temp_file.name
    return temp_file_path

def load_from_temp_file(file_path: str) -> Any:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {file_path}")
        return None


def run_prompts_in_parallel(*prompts: str) -> Dict[str, Any]:
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(prompts), os.cpu_count() or 1)) as executor:
        future_to_prompt = {executor.submit(process_prompt, prompt): prompt for prompt in prompts}
        for future in as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            try:
                result = future.result()
            except Exception as exc:
                logging.error(f"Error processing prompt: {exc}")
                result = {"error": str(exc)}
            results[prompt] = result
    return results

def process_prompt(prompt: str) -> Dict[str, Any]:
    try:
        result = a.final(prompt)
        return json.loads(result)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON from response"}
        else:
            return {"error": "No valid JSON found in response"}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.pdf'):
        try:
            extracted_data, file_path = input_pdf_setup(file)
            extracted_data_file = save_to_temp_file(extracted_data)
            session['extracted_data_file'] = extracted_data_file
            session['file_path'] = file_path
            return jsonify({"success": "File uploaded and processed successfully"})
        except Exception as e:
            logging.error(f"Error processing uploaded file: {str(e)}")
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Please upload a PDF file"}), 400

@app.route('/submit', methods=['POST'])
def submit_form():
    job_description = request.form.get('jobDescription')
    experience = request.form.get('experience')
    additional_info = request.form.get('additionalInfo')
    
    if not experience:
        return jsonify({"error": "Please select an experience level."}), 400

    if not job_description:
        return jsonify({"error": "Job description is required."}), 400

    if 'extracted_data_file' not in session:
        return jsonify({"error": "Please upload a resume first."}), 400

    try:
        extracted_data = load_from_temp_file(session['extracted_data_file'])
        if extracted_data is None:
            return jsonify({"error": "Failed to load extracted data. Please try uploading your resume again."}), 500

        session['job_description'] = job_description
        session['experience'] = experience
        session['additional_info'] = additional_info

        question = ques(extracted_data, job_description, additional_info)
        q = a.final(question)

        questions = extract_between_asterisks(q)
        questions_file = save_to_temp_file(questions)
        session['questions_file'] = questions_file
        
        return jsonify({"redirect": url_for('questionnaire', step=1)})
    except Exception as e:
        logging.error(f"Error in submit_form: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/questionnaire/<int:step>', methods=['GET', 'POST'])
def questionnaire(step: int):
    inputs = session.get('inputs', {})
    questions_file = session.get('questions_file')
    questions = load_from_temp_file(questions_file)

    if request.method == 'POST':
        inputs[f'input{step}'] = request.form.get('input_value')
        session['inputs'] = inputs
        if step < len(questions):
            return redirect(url_for('questionnaire', step=step + 1))
        else:
            return redirect(url_for('result'))
    
    if step > len(questions):
        return redirect(url_for('result'))
    
    question = questions[step - 1] if questions else "No question available"
    return render_template('input_step.html', step=step, question=question, inputs=inputs, questions=questions)

@app.route('/result')
def result():
    try:
        # Retrieve data from session
        extracted_data_file = session.get('extracted_data_file')
        if not extracted_data_file or not os.path.exists(extracted_data_file):
            flash("Resume data not found. Please upload your resume again.", 'error')
            return redirect(url_for('index'))

        extracted_data = load_from_temp_file(extracted_data_file)
        if extracted_data is None:
            flash("Failed to load resume data. Please try again.", 'error')
            return redirect(url_for('index'))

        job_description = session.get('job_description', '')
        experience = session.get('experience', 'No Experience')
        additional_info = session.get('additional_info', '')
        inputs = session.get('inputs', {})
        
        questions_file = session.get('questions_file')
        if not questions_file or not os.path.exists(questions_file):
            flash("Question data not found. Please start over.", 'error')
            return redirect(url_for('index'))

        questions = load_from_temp_file(questions_file)
        if questions is None:
            flash("Failed to load questions. Please try again.", 'error')
            return redirect(url_for('index'))


        # Prepare other questions
        other_questions = {questions[i]: inputs.get(f'input{i+1}', '') for i in range(len(questions))}
        
        # Determine if user has experience
        has_experience = experience.lower() == "experience"

        # Generate prompts based on experience
        generator_func = profile_generator_experience if has_experience else profile_generator_prompt_no_experience
        prompts = generator_func(extracted_data, job_description, additional_info, other_questions)

        # Run prompts in parallel
        results = run_prompts_in_parallel(*prompts)

        # Prepare data for template
        template_data = {
            'job_description': job_description,
            'profile_creator_result': results.get(prompts[0], {}),
            'technical_skills_result': results.get(prompts[1], {}),
            'soft_skills_result': results.get(prompts[2], {}),
            'project_result': results.get(prompts[-1], {}),
            'experience_result': results.get(prompts[3], {}) if has_experience else {}
        }

        return render_template('result.html', **template_data)

    except Exception as e:
        logging.error(f"Error in result route: {str(e)}", exc_info=True)
        flash(f"An error occurred: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/edit/<int:step>')
def edit_input(step: int):
    return redirect(url_for('questionnaire', step=step))

if __name__ == '__main__':
    app.run(debug=True, port=3313)