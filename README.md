# About
DateGen is a tool used to generate a dating profile (an introduction bio) for a person based on their attributes, which include age, sex, social habbits, et cetera.

# How
DateGen uses an LSTM network, with a single hidden layer to achieve this. The model was trained on a dataset of around 60k OkCupid users, found here - https://github.com/rudeboybert/JSE_OkCupid.

# Progress
Currently, the generated text isn't coherent enough. Future work involves tuning the hyperparameters to improve the model. Also, the network was trained for only 5 epochs. Maybe this number can be increased too. I'm also considering utilizing GPT-2 to achieve better results.

# Code
This is a Django codebase which includes the model code too. The model is made up of 4 files (under online_dating):
configs.py (Config info)
dp_gen.py (Training code)
dp_gen_web.py (Profile generator)
model.py (LSTM model)

## How To Run
If you have Django setup locally, running the following command from this repo's local directory, should work -
```
python manage.py runserver
```
NOTE 1: The python version should at least be 3.6.  
NOTE 2: The code has index.html file - move it to the django directory which looks something like this:  
        <prefix>/lib/python3.6/site-packages/django/contrib/admin/templates/index.html.  
        This is where python libraries get installed.

# Examples
If I find the time and money, I'll serve this model through a webpage. But for now, I only have some screenshots (taken from a local instance):
