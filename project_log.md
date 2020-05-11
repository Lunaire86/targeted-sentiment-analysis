## Daily Plan

### Training embeddings - NoReC
- CBoW 300  
    `sbatch norec.slurm --train /cluster/work/users/marispau/train --save /cluster/work/users/marispau/exam/embeddings --cbow --dim 300`  
- CBoW 100  
    `sbatch norec.slurm --train /cluster/work/users/marispau/train --save /cluster/work/users/marispau/exam/embeddings --cbow`
- SG 300  
    `sbatch norec.slurm --train /cluster/work/users/marispau/train --save /cluster/work/users/marispau/exam/embeddings --sg --dim 300`  
- SB 100  
    `sbatch norec.slurm --train /cluster/work/users/marispau/train --save /cluster/work/users/marispau/exam/embeddings --sg`  

### Training embeddings - NDT - blog
- CBoW 100  
    `sbatch coleus.slurm --source ../ndt/blog_ndt/ --target embeddings/ --dim 100 --epochs 10 --cbow -D --zip`

### Training models
- norec.300.cbow.bin -> 2020-05-10_02-55-11  
    - Binary(precision=0.7627, recall=0.0354, f1=0.0677)  
    - Proportional(precision=0.661, **recall=0.0307**, f1=0.0587)  
    `Namespace(batch_size=100, dropout=0.01, embeddings='norec.300.cbow.bin', embeddings_dim=300, epochs=50, hidden_dim=100, learning_rate=0.01, load=False, num_layers=1, run='baseline', saga=True, train_embeddings=False)`  
    
- norec.100.cbow.bin -> 2020-05-10_02-54-50  
    - Binary(precision=0.8235, recall=0.0331, f1=0.0636)  
    - Proportional(precision=0.6667, recall=0.0268, f1=0.0515)  
    `Namespace(batch_size=100, dropout=0.01, embeddings='norec.100.cbow.bin', embeddings_dim=100, epochs=50, hidden_dim=100, learning_rate=0.01, load=False, num_layers=1, run='baseline', saga=True, train_embeddings=False)`  
    
- norec.300.sg.bin -> 2020-05-10_02-54-42  
    - Binary(**precision=0.8378**, recall=0.0244, f1=0.0474)  
    - Proportional(**precision=0.7838**, recall=0.0228, f1=0.0444)  
    `Namespace(batch_size=100, dropout=0.01, embeddings='norec.300.sg.bin', embeddings_dim=300, epochs=50, hidden_dim=100, learning_rate=0.01, load=False, num_layers=1, run='baseline', saga=True, train_embeddings=False)`  
    
- norec.100.sg.bin -> 2020-05-10_02-41-51  
    - Binary(precision=0.6774, recall=0.0165, f1=0.0323)  
    - Proportional(precision=0.6129, recall=0.015, f1=0.0292)  
    `Namespace(batch_size=100, dropout=0.01, embeddings='norec.100.sg.bin', embeddings_dim=100, epochs=50, hidden_dim=100, learning_rate=0.01, load=False, num_layers=1, run='baseline', saga=True, train_embeddings=False)`  

### Tuesday 05.05.20
- Rebase master onto experimental (I think?)
- Write stuff about baseline model evaluation results.
- Write background stuff.
- Maybe mostly write instead of code?
- Class (or not?) + individual session with lecturer.

### Wednesday
- Group session at 12:15.
- Ideally have a different architecture up and running. 
- Or, work on training fastText embeddings on Norsk Aviskorpus + NoReC?

---

### Wednesday 29.04.20
- Get out of the rabbit hole I entered last night: sort out paths to whatever files are needed.
- Ask any TensorFlow related questions during the group lecture.
- Read at least one article and add to .bib file.
- Not fuck about, and actually get baseline model up and running.
- Check that all pickles are good! (right now they're lost in space somewhere)

### Thursday 30.04.20
- Finalise baseline model (hopefully) and describe some of the results. Perhaps link it to the projects particular «problem space», namely joint predictions of target extraction and polarity decision. Check binary F1 score against the 0.4 score mentioned during Tuesday's Q&A session.
- Smooth out any remaining issues with the data pipeline.

### Friday 01.05.20
- In the project report, start motivating the choice of experimental directions. Refer to interesting sources.
- In case `focus == None`, (Jamie will be home, after all), take a day off from code and try to focus on reading and writing report related stuff.
- Baseline model HAS TO BE READY!!!

### Saturday 02.05.20
- Let's be realistic. There won't be enough time for me to not work over the weekend...
- Start working on one of the three experiments.
- Try to make a decision about what all three directions will be. At least 2/3 of them.
- Ideally have like one page of the report ready, maybe? Introduction and background and so on.

### Sunday 03.05.20
- Read about initialisers if I haven't already.
- Be awesome.
- Go for a walk or something.

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
