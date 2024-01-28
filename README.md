# chatter

This requires a ```.env``` file, which should look similiar to the example below.

```
TEXT_GENERATION=<openai for openai, local otherwise>
OPENAI_API_KEY=<your API key>
OPENAI_ORG_ID=<your organization ID>
```

# Instructions
- Move into the ```flaskapp``` folder and install the requirements.
```
cd flaskapp
python -m venv myenv             # create a virtual environment
source myenv/bin/activate        # active the virtual environment in Linux
pip install -r requirements.txt  # install the requirements
```
- Start the flask application.
```
python app.py
```
- Move into the ```webapp``` folder and install the requirements.
```
cd webapp
npm install
```
- Start the web application.
```
npm start
```

# Misc
- Update all requirements.
```
pip install pip-upgrader
pip-upgrade requirements.txt
```
- Generate the ```requirements.txt``` file.
```
pip freeze > requirements.txt
```