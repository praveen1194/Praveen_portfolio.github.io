---
layout: post
title: Project Management Assistant
category: LLM Prompts
---

## An emailing Assistant that helps with project management

Another project where I wanted to explore using Open AI API to build a solution for assisting with project management work. I wanted to explore providing a small set of infromation (such as a company internal infromation, not available on the internet) to get a better response. I also wanted to improve this by providing the LLM model with tools at its disposal which it can use whenever it needs to (such as ability to asking for more information).

### Objectives:
1.   Assist project manager with stakeholder engagement by automating sending of emails.
2.   The project manager will provide the requirements and the software should automatically send a professional email optimsing the project managers time.
3.   Ask any question required and get all the details from the user before sending the email.

## A simple prompt for drafting email
A very simple instruction prompt gives information and context about the activity to be performed. However, there are a lot more details that can be added as shown in the next section.


```python
from google.colab import userdata
from openai import OpenAI

client = OpenAI(api_key=userdata.get("OPENAI_API_KEY"))

def generate_response(prompt, model="o4-mini"):
  response = client.responses.create(
      model= model,
      instructions = "We recently had a small glich in a project where the client forgot to send some data they promised to share on which the project work is dependent. While writing an email be polite, professional but show a sense of urgency",
      input = prompt
      );


  return response.output_text
```


```python
print(generate_response("Write an email to my client explaining the situation and ask for the data to avoid any delay."))
```

### Here is what the LLM generated

    Subject: Request for Outstanding Data to Keep [Project Name] on Schedule
    
    Dear [Client Name],
    
    I hope you’re doing well.
    
    As we move forward with [Project Name], I wanted to touch base regarding the [specific data/files—e.g., “site traffic analytics” or “vendor cost breakdown”] that you kindly offered to share. Our team is ready to proceed with the next phase, but we’re currently unable to advance until we receive this information.
    
    To ensure we stay on track for our agreed-upon milestones, could you please send the outstanding data by [desired date, e.g., “end of week” or “June 20”]? Having it in hand by then will help us:
    
    - Complete [next task] without delay  
    - Maintain our overall project timeline  
    - Deliver your expected results on schedule  
    
    If there’s any difficulty in gathering the data or if you need assistance formatting it, please let me know right away. We’re happy to help in any way we can to keep things moving smoothly.
    
    Thank you for your prompt attention to this request. I appreciate your support and look forward to receiving the materials soon.
    
    Best regards,
    
    [Your Name]  
    [Your Title]  
    [Your Company]  
    [Phone Number] | [Email Address]
    

As can be seen the email draft generated is missing a lot of details. Some information is assumed as well. Overall, the language is professional but due to lack of context and data the content is not to the level which can be automatically sent.


---



## Emailing assistant with optimisations based on the learnings above

A detailed instruction providing the LLM information about what to focus on, how to frame the email and context infromation behind the activity (Project plan of the project)

This project information is crucial for the LLM to generate a more appropriate email, however this can be done in many other ways such as RAG or providing a file with all the project details. For the demonstration purpose the most simplest approach was taken.


```python
instructions = '''
# Identity
You are an experienced project manager. You need to engage with project stakeholders in a polite and professional but firm manner and ensure all dependencies are resolved, your software development team has everything available that they need and the project executes smoothly with delivery on time.

# Instructions
* Use british english as your clients are british.
* If you need any infromation ask for it (such as any details on the context of the conversation you are asked to do) the user. Do not assume anything.
* If you need to ask any questions to the user, you MUST use the get_multiline_input tool provided for that.
* After you have collected all the information you require, write email to clients or other stake holders based on the prompt.
* While writing the email, keep it short and simple (maximum 500 words)
* The email should be well organised and use bullet points to organise information if required for easy readability.
* In order to send the email use the send_email tools provided to you.

# Context
You work in a company called SoftSolution that provides custom software solutions for clients. You are working on a project as a project manager. Here are the details of the project,

## Project: SmartDocs Automation Platform

## Client: Acme Legal Solutions Inc.

### Duration: July 1, 2025 – December 31, 2025 (6 months)

---

## Objective

Develop a custom web-based document automation platform to help Acme Legal Solutions generate, manage, and review legal documents efficiently.

---

## Key Features

- Secure user login
- Document template builder
- Smart form input with auto-fill
- Document version control
- PDF export & digital signature
- Admin dashboard with role-based access
- Integration with existing CRM via API

---

## Project Team

| Role               |
|--------------------|
| Project Manager    |
| Frontend Developer |
| Backend Developer  |
| UI/UX Designer     |
| QA Engineer        |
| DevOps Engineer    |

---

## 6-Month Timeline

| Month       | Phase                   | Key Activities |
|-------------|--------------------------|----------------|
| **July**    | Initiation & Planning     | - Requirement gathering<br>- Create wireframes<br>- Finalize tech stack |
| **August**  | Design & Setup            | - UI/UX design approval<br>- Setup dev & staging environments |
| **September** | Core Development I     | - User authentication<br>- Template editor<br>- Backend setup |
| **October** | Core Development II       | - Smart form logic<br>- CRM API integration<br>- Admin dashboard |
| **November** | Testing & QA             | - Functional & regression testing<br>- UAT with client<br>- Bug fixing |
| **December** | Deployment & Handover    | - Production deployment<br>- Documentation<br>- Team training & support |

---

## Client Dependencies

To ensure timely delivery and success of the project, the following inputs and actions are required from Acme Legal Solutions:

- **Sample Legal Documents**: Representative templates and forms to be automated (due by July 10)
- **CRM API Documentation & Access**: Technical documentation and credentials for integration (due by August 15)
- **Design Feedback**: Timely feedback on UI/UX prototypes (within 3 business days of submission)
- **Test Users**: Access to a small group of internal users for UAT in November
- **Single Point of Contact**: A designated stakeholder for rapid decision-making and clarification

---

## Deliverables

- Fully deployed web application
- Technical and user documentation
- Client training session
- 1-month post-deployment support

---
'''
```

This is a very simple approach to obtain information from user - A new input is created everytime user presses enter untill the user types ***END***.

This method for taking input from user is crude because this is done on a google colab notebook and does not have a lot of tools to define a proper interface.

**Note:-** I tried using *ipywidgets* but it was not possible to extract information from the `onButtonClick()` event since it executes asynchronously. I tried using options like `nonlocal` and `global` variables as well as `asyncio` libraries but that still did not work as I expected. There might be a better way but since UI was not the main objetive I did not invest more time in it.


```python
def get_multiline_input(question):
  '''
  creates a multuline input allowing the LLM to ask questions form the user

  Parameters:
    question - multiline string with all the questions (each in a single line) that the LLM needs to ask the user

  returns:
    multiline string, which is the answer provided by the user
  '''
  print(question)

  print("\n Enter your text (type 'END' on a new line to finish):\n")
  lines = []
  while True:
      line = input()
      if line.strip().upper() == 'END':
          break
      lines.append(line)
  return '\n'.join(lines)
```


```python
def send_email(recipients, subject, body):
  '''
  Sends email to the recipients with the provided subject and body

  Parmeters:
    recipients - email ids of the recipients of the email
    subject - subject of the email
    body - body of the email

  Ouput:
    sends an email
  '''
  # integrate API with SMTP server credentials here to actually send emails. This is an example of how it can be done.

  print(recipients)
  print(subject)
  print(body)
```

Both the functions defined above for taking user input and sending email are provided as tools for the LLM.

Here, `tool_choice = "required"` in the request becasue the LLM will either take user input or send email. In no scenario the assistant will **NOT** call a tool (function).

### Bringing it all together


```python
import json

def generate_response(prompt, model="o4-mini"):
  '''
  Calls the Open AI API to send email based on the user prompt

  Parameters:
    prompt - user prompt for the LLM

  Output:
    sends an email after asking clarifying questions from the user
  '''

  tool = [
      {
      "type": "function",
      "name": "get_multiline_input",
      "description": "Gets input from the user for any questions that the assistant might need to ask from the user",
      "strict": True,
      "parameters": {
          "type": "object",
          "properties": {
              "questions": {"type": "string",
                            "description": "all the questions that the assistant needs to ask from the user in multiline(one question on each line)"}
          },
          "required": ["questions"],
          "additionalProperties": False
          }
      },
      {
      "type": "function",
      "name": "send_email",
      "description": "Sends emails.",
      "strict": True,
      "parameters": {
          "type": "object",
          "properties": {
              "recipients": {"type": "string",
                            "description": "all the recipients whom the email needs to be sent to seperated by comma (',')"},
              "subject": {"type": "string",
                            "description": "Subject of the email"},
              "body": {"type": "string",
                            "description": "Body of the email"}
          },
          "required": ["recipients", "subject", "body"],
          "additionalProperties": False
          }
      }
  ]

  input_messages = [{"role": "user", "content": prompt}]

  response = client.responses.create(
      model= model,
      instructions = instructions,
      input = input_messages,
      tools = tool,
      tool_choice = "required"
      );

  while(True):
    for item in response.output:
      if item.type == "function_call" and item.name == "get_multiline_input":
        args = json.loads(item.arguments)
        user_response = get_multiline_input(args["questions"])

        input_message = [{"type": "function_call_output",
                               "call_id": item.call_id,
                               "output": str(user_response)
                               }]

        response = client.responses.create(
            model= model,
            instructions = instructions,
            input = input_message,
            previous_response_id = response.id,
            tools = tool,
            tool_choice = "required"
            );

      elif item.type == "function_call" and item.name == "send_email":
        args = json.loads(item.arguments)
        send_email(args["recipients"], args["subject"], args["body"])
        return
```


```python
generate_response("We were expecting an input from the client which is a dependency and they haven't replied for. This is going to hold off the work if they don't provide that soon. Send an email to my client explaining the situation and ask for the data to avoid any delay.")
```

### This was the response from the LLM

    Which specific client input are we awaiting? Please provide the input name and its original due date.
    What is the name and email address of the single point of contact at Acme Legal Solutions for this input?
    
     Enter your text (type 'END' on a new line to finish):
    
    We are waiting for design feedback. It was due by 15 August, 2025
    Name - Person1, email - person1@acme.com
    END
    person1@acme.com
    Reminder: Design Feedback for SmartDocs Automation Platform
    Dear Person1,
    
    I hope you’re well. I’m writing to follow up on the design feedback for the SmartDocs Automation Platform, which was due on 15 August 2025. As this input is critical to the UI/UX design approval phase, we’re currently unable to progress with the design and setup without it.
    
    To avoid any impact on our project timeline (July–December 2025), please could you provide your feedback by the end of this week? Specifically:
    
    • Any comments or revisions on the submitted wireframes and mock-ups
    • Approval or suggested changes to the colour scheme, typography and layout
    • Any other general design considerations or constraints
    
    If you require additional information or would like to schedule a quick call to discuss, please let me know. Your prompt response will help ensure we remain on track for delivery.
    
    Thank you for your attention to this matter.
    
    Best regards,
    
    [Your Name]
    Project Manager
    SoftSolution Ltd.
    your.email@softsolution.com
    +44 20 7123 4567
    
