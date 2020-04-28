# So reminder. Much work. TODO.
The idea is that I can use this as a project log. Maybe talk to myself a little when I've worked for 12 hours and feel a bit weird. Keep myself in check. Did I spend several hours yesterday balls deep in regex patterns? Yes. Was that fruitful? Also yes. Kind of. Maybe only a little. Stop nagging.

28.04.20 - Agenda
---
1. Baseline...
2. Track meeting 12:00.
3. Read the relevant chapter(s) in the Neural Methods book.
4. Read the relevant chapter(s) in Jurafsky and Martin.
5. Create the daily plan.

27.04.20 - Agenda
---
1. Get the baseline model up and running.
2. Get a LaTeX skeleton ready and check that it compiles.
3. Create a table with a daily plan in this log document.

##### Look into whether it's possible and/or useful to experiement with any of the following:

**kernel_initializer**  
– Initializer for the `kernel` weights matrix, used for the linear transformation of the inputs. Default: `glorot_uniform`.  
**recurrent_initializer**  
– Initializer for the `recurrent_kernel` weights matrix, used for the linear transformation of the recurrent state. Default: `orthogonal`.   
**bias_initializer**  
– Initializer for the bias vector. Default: `zeros`.  

26.04.20 - Agenda
---
1. Vectorise dataset (from words to integers)
2. Finalise dataset (from integer sequences of variable length, to padded floating point vectors)
3. Build and compile the BiLSTM baseline model
4. Train model on training data
5. Run the evaluation scripts on dev data
6. Run both the PyTorch pre-code model, and my TensorFlow model, on Saga. Compare differences.
7. Zoom meeting with Erling.

«You better sort this shit out, or else...»
---
- PLZ get the baseline model up and running ASAP!!!!!
- Make sure the data is pre-processed the same way.
- Remove hardcoded paths from files on git... >_<

«If I may make a suggestion...»
---
- Edit README.md to reflect actual project structure.
- Edit .gitignore
- Possibly ask someone to make sure you've not fucked up the baseline model somehow. Because that would suck. Hard.

«It's totally up to you, like...»
--- 
- Go the Sphinx way for docs and shit. 
- Update the environment.yml file with the correct libraries.
- Set up with Makefile etc.

Bro, do you even read this far?
---
- Yes, as a matter of fact, I do.
- Remove commented-out code in WordEmbeddings.pickle()
- Make nicer plots of shit in R.
