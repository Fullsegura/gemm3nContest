This the server component of "Guardians of Mother Earth" platform.
It includes:
- Gemma 3n server, gemmaKagle.py. It is an API receiving prompt an image as input. they are pass to Gemma 3n local model to send back contextual reasoned answers.
- Python server, mainKagle.py. It is the process connected to Firebase to update in real time questions and answers from APP Internet queries. 