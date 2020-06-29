import neptune


"""
What's in the code?
﻿init() - set project

﻿create_experiment() - creates new experiment in the project - it is our gateway to Neptune

﻿log_metric() - logs numeric values to neptune 

﻿stop() - stop and close the experiment

"""

neptune.init('davquispe/sandbox')
neptune.create_experiment(name='minimal_example')

for i in range(100):
    neptune.log_metric('loss', 0.6**i)

neptune.log_metric('AUC', 0.96)




"""
You define your parameters at experiment creation.

To run a new experiment with defined parameters, add to your script this sample snippet:
"""

# Define parameters

PARAMS = {'decay_factor' : 0.5,
          'n_iterations' : 117}

# Create experiment with defined parameters

neptune.create_experiment (name='example_with_parameters',
                          params=PARAMS)


"""
Log images

The image can be PIL.Image, matplotlib figure, numpy array (2d or 3d) or path to an image.
To log image data to Neptune, use the neptune.log_image() method.

"""

# Log image data

import numpy as np

array = np.random.rand(10, 10, 3)*255
array = np.repeat(array, 30, 0)
array = np.repeat(array, 30, 1)
neptune.log_image('mosaics', array)

"""
Log text

Log a single line of text to the current experiment as a Python string using neptune.log_text().
"""
# Log text data

neptune.log_text('top questions', 'what is machine learning?')

"""
Log Artifacts
There are many things that you might want to save during training that are neither metrics nor images. 
Neptune allows you to log any file as an artifact. That means you can save your model checkpoint or results as a .pkl file and log that to the app.

To save an artifact (file) to Neptune experiment storage, use the neptune.log_artifact() method, when the experiment is running. 
Note that file is being uploaded from the local machine.

"""
# log some file

# replace this file with your own file from local machine
neptune.log_artifact('flow_xtrain_15kHz.pkl')  

# log file to some specific directory (see second parameter below)

# replace this file with your own file from local machine
#neptune.log_artifact('model_checkpoints/checkpoint_3.pkl', 'training/model_checkpoints/checkpoint_3.pkl')


"""
You might want to track your codebase too. Neptune allows you to upload your code and preview in the UI. 
Just choose the files that you want to send to Neptune.
"""

# Upload source code

# replace these two source files with your own files.
neptune.create_experiment(upload_source_files=['example_neptune.py', 'KMeans_leak_vs_noleak_functions.py']) 


"""
Adding tags
"""

# add tag when experiment is created
neptune.create_experiment(tags=['training'])

# add single tag
neptune.append_tag('transformer')

# add few tags at once
neptune.append_tags('BERT', 'ELMO', 'ideas-exploration')