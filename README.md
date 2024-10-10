# SubjectChatbot_Project
This is the dedicated project by five people on the subject Tutor Chatbot

# How to run this project like a pro
(cd to the BE_chatbot file)
1. Create a virtual len environment (Always open this backend terminal in git bash):
 + install virtual len: **pip install virtualenv**
 + create the environment: **virtualenv venv**
 + activate the environment: **source venv/Scripts/activate** (If it have the (venv) above the bash -> successfully activate the environment, if not then get good!)
2. Install the requirement file while activating the environment: **pip install -r requirements.txt**
3. Run the Chatbot logic: **python app.py**
4. You need to create a file .env package BE_chatbot 
```bash
export GEMINI_API_KEY = 'AIzaSyAMiTwhDIYIOb-mkNAgY_2EHx1KgIhKqMY'
export PORT = 3001


(cd to the BE_Nodejs -> be-v1)
1. Install the node module: **npm install**
2. Start the database: **npm start**

(cd to the FE -> fe-v1)
1. Install the node module: **npm install**
2. Start the react front end: **npm start**

# Disclaimer: when running the project, always run like this position BE_chatbot -> be-v1 -> fe-v1
