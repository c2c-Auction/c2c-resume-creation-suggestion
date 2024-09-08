import logging
from typing import Dict, Any, List, Tuple
from flask import Flask, request, render_template, redirect, url_for, flash, session
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
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


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
def input_pdf_setup(uploaded_file, api_key) -> Tuple[str, str]:
    try:
        # Ensure the filename is secure
        filename = uploaded_file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the uploaded file to the specified path
        uploaded_file.save(filepath)
        
        # Load the PDF file using PyPDFLoader
        loader = PyPDFLoader(file_path=filepath)
        
        # Split the PDF into pages
        pages = loader.load_and_split()
        
        # Check if the PDF has no pages
        if len(pages) < 1:
            raise ValueError("The PDF file has no pages.")
        
        # Extract text content from the pages
        text = " ".join([page.page_content for page in pages])
        
        # Pass the api_key to a.final()
        return a.final(text, api_key), filepath
    
    except FileNotFoundError:
        raise FileNotFoundError("The specified file was not found.")
    
    except IOError:
        raise IOError("An I/O error occurred while handling the PDF file.")
    
    except Exception as e:
        # Catch any other exceptions and raise them with a custom message
        raise Exception(f"An unexpected error occurred: {str(e)}")
    

def save_to_temp_file(data: Any) -> str:
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
        json.dump(data, temp_file, ensure_ascii=False)
    return temp_file.name

def load_from_temp_file(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def run_prompts_in_parallel(*prompts: str, api_key: str) -> Dict[str, Any]:
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(prompts), os.cpu_count() or 1)) as executor:
        future_to_prompt = {executor.submit(process_prompt, prompt, api_key): prompt for prompt in prompts}
        for future in as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            try:
                result = future.result()
            except Exception as exc:
                logging.error(f"Error processing prompt: {exc}")
                result = {"error": str(exc)}
            results[prompt] = result
    return results

def process_prompt(prompt: str, api_key: str) -> Dict[str, Any]:
    try:
        result = a.final(prompt, api_key)  # Pass the API key to the final function
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        file = request.files.get('file-upload')
        job_description = request.form.get('jobDescription')
        experience = request.form.get('experience')
        additional_info = request.form.get('additionalInfo')
        groq_api_key = request.form.get('groq_api_key')

        # Process the resume
        try:
            resume_text, filepath = input_pdf_setup(file, groq_api_key)  # Pass groq_api_key here
            # Construct the input for the final function
            input_text = f"Resume: {resume_text}\nJob Description: {job_description}\nAdditional Info: {additional_info}\nExperience: {experience}"
            result = a.final(input_text, groq_api_key)  # Pass the API key to the final function

            # Store necessary data in session for later use
            session['extracted_data_file'] = save_to_temp_file(resume_text)
            session['job_description'] = job_description
            session['experience'] = experience
            session['additional_info'] = additional_info
            session['groq_api_key'] = groq_api_key

            # Generate questions
            questions_prompt = ques(resume_text, job_description, additional_info)
            questions_result = a.final(questions_prompt, groq_api_key)
            questions = extract_between_asterisks(questions_result)
            session['questions_file'] = save_to_temp_file(questions)

            return redirect(url_for('questionnaire', step=1))

        except Exception as e:
            flash(f"An error occurred: {str(e)}", 'error')
            return render_template('index.html', error=str(e))

    return render_template('index.html')

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
        extracted_data = load_from_temp_file(session.get('extracted_data_file'))
        job_description = session.get('job_description', '')
        experience = session.get('experience', 'No Experience')
        additional_info = session.get('additional_info', '')
        inputs = session.get('inputs', {})
        questions = load_from_temp_file(session.get('questions_file'))
        groq_api_key = session.get('groq_api_key')  # Retrieve API key from session

        # Prepare other questions
        other_questions = {questions[i]: inputs.get(f'input{i+1}', '') for i in range(len(questions))}
        
        # Determine if user has experience
        has_experience = experience.lower() == "experience"

        # Generate prompts based on experience
        generator_func = profile_generator_experience if has_experience else profile_generator_prompt_no_experience
        prompts = generator_func(extracted_data, job_description, additional_info, other_questions)

        # Run prompts in parallel
        results = run_prompts_in_parallel(*prompts, api_key=groq_api_key)  # Pass API key here

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

# Add this at the end of the file
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

if __name__ == '__main__':
    app.run()