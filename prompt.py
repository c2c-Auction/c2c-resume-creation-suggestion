def profile_generator_experience(user_resume, job_description, work, other_questions):
    profile_creator_prompt = f"""
    
    Return the result in JSON format:
    {{
    "summary": "<Your profile summary here>",
    "Reasonof_output_matching": "<Your reason here>"
    }}
    Your task is to create a brief but impactful profile summary based on the provided details, limiting it to 50-100 words. Focus on the user's most relevant skills, experiences, and achievements that align with the job requirements. Mention any unique qualities or experiences that distinguish the user from other candidates.

    Input Details:
    User Resume: {user_resume}
    Job Description: {job_description}
    Additional Work Experience: {work}
    Answers to Specific Questions: {other_questions}

    Return the result in JSON format:
    {{
      "summary": "<Your profile summary here>",
      "Reasonof_output_matching": "<Your reason here>"
    }}
    """
    
    technical_skills_prompt = f"""
     Return the result in JSON format:
    {{
      "technical_skills": ["<Your technical skill 1>", "<Your technical skill 2>", ...],
      "Reasonof_output_matching": "<Your reason here>"
    }}
    Your task is to list technical skills based on the provided details, limiting it to 10-20 keywords. Focus on the user's most relevant skills, experiences, and achievements that align with the job requirements. Mention any unique qualities or experiences that distinguish the user from other candidates.

    Input Details:
    User Resume: {user_resume}
    Job Description: {job_description}
    Additional Work Experience: {work}
    Answers to Specific Questions: {other_questions}

    Return the result in JSON format:
    {{
      "technical_skills": ["<Your technical skill 1>", "<Your technical skill 2>", ...],
      "Reasonof_output_matching": "<Your reason here>"
    }}
    """
    
    soft_skills_prompt = f"""
    Return the result in JSON format:
    {{
      "soft_skills": ["<Your soft skill 1>", "<Your soft skill 2>", ...],
      "Reasonof_output_matching": "<Your reason here>"
    }}
    Your task is to list soft skills based on the provided details, limiting it to 10-20 keywords. Focus on the user's most relevant skills, experiences, and achievements that align with the job requirements. Mention any unique qualities or experiences that distinguish the user from other candidates.

    Input Details:
    User Resume: {user_resume}
    Job Description: {job_description}
    Additional Work Experience: {work}
    Answers to Specific Questions: {other_questions}

    Return the result in JSON format:
    {{
      "soft_skills": ["<Your soft skill 1>", "<Your soft skill 2>", ...],
      "Reasonof_output_matching": "<Your reason here>"
    }}
    """
    experience_prompt = f"""

     Return the result in JSON format:
     {{
   experiences: ["<company name - Role Name 1>", "<company name - Role Name 2>", ...],
   experience_details: ["<Your experience detail 1 (3-5 lines)>", "<Your experience detail 2 (3-5 lines)>", ...],
   Reasonof_output_matching: "<Your reason here>"
    }}
    Your task is to describe experiences based on the provided details, limiting each experience to 5-10 lines . Focus on the user's most relevant skills, experiences, and achievements that align with the job requirements. Mention any unique qualities or experiences that distinguish the user from other candidates.

    Guidelines for the Experience Paragraph:
    - Introduction: Start with a brief mention of your current or most recent role and the company name.
    - Key Responsibilities: Highlight 1-2 primary responsibilities that are most relevant to the job you are applying for.
    - Achievements: Mention 1-2 significant accomplishments or projects that demonstrate your impact and skills.
    - Skills Applied: Optionally, you can mention specific skills or technologies used in your achievements.

    Input Details:
    User Resume: {user_resume}
    Job Description: {job_description}
    Additional Work Experience: {work}
    Answers to Specific Questions: {other_questions}

     Return the result in JSON format:
     {{
   experiences: ["<company name - Role Name 1>", "<company name - Role Name 2>", ...],
   experience_details: ["<Your experience detail 1 (3-5 lines)>", "<Your experience detail 2 (3-5 lines)>", ...],
   Reasonof_output_matching: "<Your reason here>"
    }}
    """

    
    project_prompt = f"""
        Return the result in JSON format:
       {{
   project: ["<Your project 1>", "<Your project 2>", ...],
   project_details: ["<Your project detail 1>", "<Your project detail 2>", ...],
   Reasonof_output_matching: "<Your reason here>"
    }}
    Your task is to describe projects based on the provided details, limiting each project to 3-7 lines. Focus on the user's most relevant skills and achievements that align with the job requirements. Mention any unique qualities or experiences that distinguish the user from other candidates.

    Guidelines for Project Descriptions:
    - Purpose: Briefly state what the project was about.
    - Role and Responsibilities: the key tasks you performed and mention the key technologies or tools used.
    - Technologies Used: Highlight the main technologies or tools you utilized.
    - Outcome/Impact: State the results or benefits achieved through the project.

    Input Details:
    User Resume: {user_resume}
    Job Description: {job_description}
    Additional Work: {work}
    Answers to Specific Questions: {other_questions}

    
     Return the result in JSON format:
       {{
   project: ["<Your project 1>", "<Your project 2>", ...],
   project_details: ["<Your project detail 1>", "<Your project detail 2>", ...],
   Reasonof_output_matching: "<Your reason here>"
    }}
    """
    
    return profile_creator_prompt, technical_skills_prompt, soft_skills_prompt, experience_prompt, project_prompt
def profile_generator_prompt_no_experience(user_resume, job_description, work, other_questions):
    
    profile_creator_prompt = f"""
     Return the result in JSON format:
    {{
      "summary": "<Your profile summary here>",
      "Reasonof_output_matching": "<Your reason here>"
    }}
    Your task is to create a brief but impactful profile summary based on the provided details, limiting it to 50-150 words. Focus on the user's most relevant skills and achievements that align with the job requirements. Mention any unique qualities or experiences that distinguish the user from other candidates.

    Input Details:
    User Resume: {user_resume}
    Job Description: {job_description}
    Additional Work Experience: {work}
    Answers to Specific Questions: {other_questions}

    Return the result in JSON format:
    {{
      "summary": "<Your profile summary here>",
      "Reasonof_output_matching": "<Your reason here>"
    }}
    """

    technical_skills_prompt = f"""
     Return the result in JSON format:
    {{
      "technical_skills": ["<Your technical skill 1>", "<Your technical skill 2>", ...],
      "Reasonof_output_matching": "<Your reason here>"
    }}
    Your task is to provide a list of technical skills based on the provided details, limiting it to 10-20 keywords. Focus on the user's most relevant skills and achievements that align with the job requirements. Mention any unique qualities or experiences that distinguish the user from other candidates.

    Input Details:
    User Resume: {user_resume}
    Job Description: {job_description}
    Additional Work Experience: {work}
    Answers to Specific Questions: {other_questions}

    Return the result in JSON format:
    {{
      "technical_skills": ["<Your technical skill 1>", "<Your technical skill 2>", ...],
      "Reasonof_output_matching": "<Your reason here>"
    }}
    """

    soft_skills_prompt = f"""
     Return the result in JSON format:
    {{
      "soft_skills": ["<Your soft skill 1>", "<Your soft skill 2>", ...],
      "Reasonof_output_matching": "<Your reason here>"
    }}
    Your task is to provide a list of soft skills based on the provided details, limiting it to 10-20 keywords. Focus on the user's most relevant skills, experiences, and achievements that align with the job requirements. Mention any unique qualities or experiences that distinguish the user from other candidates.

    Input Details:
    User Resume: {user_resume}
    Job Description: {job_description}
    Additional Work Experience: {work}
    Answers to Specific Questions: {other_questions}

    Return the result in JSON format:
    {{
      "soft_skills": ["<Your soft skill 1>", "<Your soft skill 2>", ...],
      "Reasonof_output_matching": "<Your reason here>"
    }}
    """

    project_prompt = f"""
        Return the result in JSON format:
       {{
   project: ["<Your project 1>", "<Your project 2>", ...],
   project_details: ["<Your project detail 1>", "<Your project detail 2>", ...],
   Reasonof_output_matching: "<Your reason here>"
    }}
    Your task is to describe projects based on the provided details, limiting each project to 5-10 sentences. Focus on the user's most relevant skills and achievements that align with the job requirements. Mention any unique qualities or experiences that distinguish the user from other candidates.

    Guidelines for Project Descriptions:
    - Purpose: Briefly state what the project was about.
    - Role and Responsibilities: Mention your role and the key tasks you performed.
    - Technologies Used: Highlight the main technologies or tools you utilized.
    - Outcome/Impact: State the results or benefits achieved through the project.

    Input Details:
    User Resume: {user_resume}
    Job Description: {job_description}
    Additional Work: {work}
    Answers to Specific Questions: {other_questions}

    
     Return the result in JSON format:
       {{
   project: ["<Your project 1>", "<Your project 2>", ...],
   project_details: ["<Your project detail 1>", "<Your project detail 2>", ...],
   Reasonof_output_matching: "<Your reason here>"
    }}
    """
    
    return profile_creator_prompt, technical_skills_prompt, soft_skills_prompt, project_prompt
