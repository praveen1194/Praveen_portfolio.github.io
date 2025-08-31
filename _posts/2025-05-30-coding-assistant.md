---
layout: post
title: Coding Assistant
category: LLM Prompts
---

## **A SIMPLE ASSISTANT TO GENERATE CODE FROM A COMMENT**

I wanted to explore and learn using the Open AI API. Here, I used the API to build a very simple yet powerful assistant to show how an assistant can be used in software development, reducing time to deployment with lesser resources.

### Objectives:
1.   Assist in software development.
2.   Generate code based on the user comments.
3.   Create a code which is robust, reliable and secure

## A simple prompt to generate code with a simple instruction


```python
from google.colab import userdata
from openai import OpenAI

client = OpenAI(api_key=userdata.get("OPENAI_API_KEY"))

def generate_response(prompt, model="o4-mini"):
  response = client.responses.create(
      model= model,
      instructions = "You are a highly experienced software engineer. Ask for clear clarifications if something is not clear but do not assume anything. Avoid any explaination text, write a code based on the user prompt.",
      input = prompt
      );


  return response.output_text
```


```python
print(generate_response("a python code to connect to a mysql database and insert a row into a table"))
```

### LLM generated response

    ```python
    import mysql.connector
    from mysql.connector import Error
    
    def insert_row(host, database, user, password, table, data):
        conn = None
        try:
            conn = mysql.connector.connect(
                host=host,
                database=database,
                user=user,
                password=password
            )
            cursor = conn.cursor()
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['%s'] * len(data))
            sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, tuple(data.values()))
            conn.commit()
        except Error as e:
            print(f"Error: {e}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
    
    if __name__ == "__main__":
        cfg = {
            "host": "localhost",
            "database": "mydb",
            "user": "myuser",
            "password": "mypassword"
        }
        row = {
            "column1": "value1",
            "column2": 42
        }
        insert_row(
            cfg["host"],
            cfg["database"],
            cfg["user"],
            cfg["password"],
            "mytable",
            row
        )
    ```
    

In the above the **instructions** option tries to prevent halucinations and prompts the LLM to ask for queries

> instructions = You are a highly experienced software engineer. Ask for clear clarifications if something is not clear but do not assume anything. Avoid any explaination text, write a code based on the user prompt.



---



## **Detailed zero shot prompt for a more scalable, modular and readable code than the previous response**

**Note:-** this is not built to make a code in steps rather to build a small code based on a simple comment for demonstration. Many improvement can be done in this such as creating a loop where previous requests and responses are fed back to the LLM to build code in step by step process, however this is just an example.


```python
instructions = '''
# Identity
You are an experienced software engineer who focuses on good coding practices, scalability, modularity and security in a software solution. You are going to generate good quality code based on the user comments.
Enforce the use of snake case variables.

# Instructions
* Avoid generating any explaination and conclusion text to explain the code. Generate only the code.
* Make sure to generate clear descriptive comments to explain the input and output parameters of the code. Also add comments to explain something that might be difficult to understand for an entry level software developer.
This should ensure readability, but do not write too many comments.
* Ask any clarifying questions if something is ambiguous or not clear. Do not assume anything.
* Plan extensively before generating the code and ask all questions you need together at once. Make the code scalable and modular using classes and functions for repeated code sections. Specify clearly if there is a need to segregate code in different python files.
* If you are asking a question, the response_type should be "Question", otherwise if it is your answer to the prompt then the response_type should be "Answer".
* Consider SQL injections or any other types of security concerns. The code should be secure.
* Use appropriate access levels and other OOPs concepts when using classes.
* Generate configuration files to store dynamic config parameters if required. Also provide a seperate class for reading and writing to any such config files.
* Implement appropriate error handling and logging where required.
* Use appropriate spacing, the code should be well organised and neat.
'''
```

Function to take a follow up response if the LLM needs to ask some questions from the user.


```python
from IPython.display import clear_output
import ipywidgets as widgets

def follow_up(question, previous_response, model="o4-mini"):
  '''
  Create an input for the LLM to be able to ask questions from user of needed and generate code based on the answers.

  Parameters:
    question - multiline string of all the questions the LLM needs to ask the user
    previous_response - response id of the previous request to the LLM to provide the previous context
    model - LLM model to be used

  output:
    Generated code by the LLM based on the user comment and follow up clarification answers
  '''
  print(question)

  textarea = widgets.Textarea(
      value='',
      placeholder='Enter your answer here...',
      description='Response:',
      layout=widgets.Layout(width='50%', height='100px')
  )

  button = widgets.Button(description="Submit Response")

  output = widgets.Output()

  def on_submit(b):
      user_reply = textarea.value
      with output:
        clear_output()
        print("Submitting response...")

        response = client.responses.create(
            model=model,
            previous_response_id=previous_response.id,
            input=user_reply
        )
        print("Response received:")
        print(response.output_text)

  button.on_click(on_submit)

  display(textarea, button, output)
```

# Call the LLM to get a reponse from the comment
The LLM responds in the form of a JSON using pydantic with a variable which informs wether it is a question or an answer. If it is a question then the user is prompted for a response and the LLM generates a comprehesive code from the user response.

This is one way of building an LLM tool that can take one user response if LLM needs some answers. Here is an example of the JSON structure.


If it is a question:
```
{
  "response_type": "Question",
  "response_text": "Which programming language would you like to use?"
}
```


If it is an answer:
```
{
  "response_type": "Answer",
  "response_text": "<Code based on the prompt>"
}
```

## Putting it all together


```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel

class Category(str, Enum):
    Question = "Question"
    Answer = "Answer"

class response_json(BaseModel):
  response_type: Optional[Category]
  response_text: str

def generate_response(prompt, model="o4-mini"):
  '''
  Generate either a code or follow up questions for the user.

  Parameters:
    prompt - user prompt
    model - model to be used

  output:
    follow up questions for user or a code based on the comments
  '''

  response = client.responses.parse(
      model= model,
      instructions = instructions,
      input = prompt,
      text_format = response_json
      );

  parsed_response: response_json = response.output_parsed

  if parsed_response.response_type == Category.Question:
    follow_up(parsed_response.response_text, response)
  else:
    print(response.output_parsed)
```

### Now to test the optimised implementation

```python
generate_response("a python code to connect to a mysql database and insert a row into a table")
```

### LLM generated output:

I need a few details to generate secure, modular code:
    
  1. Which MySQL client library do you prefer (e.g., mysql-connector-python, PyMySQL)?
  2. Please provide the database connection parameters or the keys you want in a config file (host, port, user, password, database).
  3. What is the target table name and its column names/types? What sample values should we insert?
  4. Do you want to store connection parameters in a config file? If so, which format (INI, JSON, YAML) and file path?
  5. Do you require connection pooling?
  6. What logging details do you need (file path, log level)?
  7. Any other specific error-handling or transaction requirements?
  
### I gave the following Response:
1. mysql-connector-python
2. host - localhost, port - 4001, user - dbUser, password - pswd, database - testDB
3. Table name - employee; column names - id(integer), name(string), role(string)
4. I want to store them in an xml, use a generic path
5. Yes i want to use connection pooling
6. use a configurable dummy filepath that can be modified later, use INFO, DEBUG and ERROR log levels
7. Use custom exceptions if needed

### Below is a output of the LLM  

1) config.xml (place in same folder or adjust path)  
```
<?xml version="1.0"?>
<config>
  <database>
    <host>localhost</host>
    <port>4001</port>
    <user>dbUser</user>
    <password>pswd</password>
    <database>testDB</database>
    <pool_name>mypool</pool_name>
    <pool_size>5</pool_size>
  </database>
  <logging>
    <log_file>./app.log</log_file>
    <level>INFO</level>
  </logging>
</config>
```

1) app.py  
```python
import xml.etree.ElementTree as ET
import logging
from mysql.connector import pooling, Error as MySQLError

# --- Custom Exceptions ---
class ConfigError(Exception):
    pass

class DatabaseConnectionError(Exception):
    pass

class DataInsertError(Exception):
    pass

# --- Configuration Loader ---
def load_config(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        raise ConfigError(f"Failed to parse config: {e}")

    def get_text(path):
        node = root.find(path)
        if node is None or not node.text:
            raise ConfigError(f"Missing config element: {path}")
        return node.text.strip()

    db_cfg = {
        'host':     get_text('database/host'),
        'port':     int(get_text('database/port')),
        'user':     get_text('database/user'),
        'password': get_text('database/password'),
        'database': get_text('database/database'),
        'pool_name':get_text('database/pool_name'),
        'pool_size':int(get_text('database/pool_size')),
    }
    log_cfg = {
        'file': get_text('logging/log_file'),
        'level': get_text('logging/level').upper()
    }
    return db_cfg, log_cfg

# --- Logger Setup ---
def setup_logger(log_file, level_str):
    level = getattr(logging, level_str, None)
    if level is None:
        raise ConfigError(f"Invalid log level: {level_str}")
    logger = logging.getLogger("app")
    logger.setLevel(level)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

# --- Database Handler ---
class Database:
    def __init__(self, cfg, logger):
        self.logger = logger
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name=cfg['pool_name'],
                pool_size=cfg['pool_size'],
                host=cfg['host'],
                port=cfg['port'],
                user=cfg['user'],
                password=cfg['password'],
                database=cfg['database']
            )
            self.logger.debug("Connection pool created.")
        except MySQLError as e:
            raise DatabaseConnectionError(f"Could not create pool: {e}")

    def insert_employee(self, emp_id, name, role):
        conn = None
        cursor = None
        try:
            conn = self.pool.get_connection()
            cursor = conn.cursor()
            sql = "INSERT INTO employee (id, name, role) VALUES (%s, %s, %s)"
            cursor.execute(sql, (emp_id, name, role))
            conn.commit()
            self.logger.info(f"Inserted employee: id={emp_id}, name={name}, role={role}")
        except MySQLError as e:
            if conn:
                conn.rollback()
            msg = f"Insert failed for id={emp_id}: {e}"
            self.logger.error(msg)
            raise DataInsertError(msg)
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

# --- Main ---
if __name__ == "__main__":
    try:
        db_cfg, log_cfg = load_config("config.xml")
        logger = setup_logger(log_cfg['file'], log_cfg['level'])
        db = Database(db_cfg, logger)
        # Example insert
        db.insert_employee(1, "Alice", "Engineer")
    except (ConfigError, DatabaseConnectionError, DataInsertError) as e:
        print(f"Fatal: {e}")
        # If you want to log to console as well:
        # logger = logging.getLogger("app")
        # logger.error(f"Fatal error: {e}")
        exit(1)
```

How it works  
1. load_config parses XML and returns two dicts (db and logging).  
2. setup_logger configures a rotating logger (single file handler here).  
3. Database initializes a mysql-connector-python connection pool.  
4. insert_employee acquires a connection, does an INSERT, commits or rolls back on error, and raises a custom exception if needed.  

You can change `config.xml` path, log file path, log level or pool size without touching the code.

