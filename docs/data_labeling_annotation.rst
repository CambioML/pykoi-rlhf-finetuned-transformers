Data Labeling & Annotation
===================================

Creating high-quality data is extremely important for building robust language applications.

.. note::

   If you have a model you'd like to collect live-feedback for, check out the Data Collection & Feedback section.

**Question & Answering**

Create labeling tasks for summarizing written documents. Users can reference a set of 
predefined summaries, for example, ...

.. code:: python

  import pykoi as pk

  # assume you have some model, endpoint, or api
  model = your_model

  # give model chat ui with feedback
  chatbot = pk.Chatbot(model)

  # Create sharable link to the application
  app = pk.Application(debug=False, share=False)
  app.add_component(chatbot)
  app.run()

img 

**Summarization**

Create labeling tasks for summarizing written documents. Users can reference a set of 
predefined summaries, for example, ...

.. code:: python

  import pykoi as pk

  # assume you have some model, endpoint, or api
  model = your_model

  # give model chat ui with feedback
  chatbot = pk.Chatbot(model)

  # Create sharable link to the application
  app = pk.Application(debug=False, share=False)
  app.add_component(chatbot)
  app.run()

img 

**Named Entity Recognition**

Create labeling tasks for summarizing written documents. Users can reference a set of 
predefined summaries, for example, ...

.. code:: python

  import pykoi as pk

  # assume you have some model, endpoint, or api
  model = your_model

  # give model chat ui with feedback
  chatbot = pk.Chatbot(model)

  # Create sharable link to the application
  app = pk.Application(debug=False, share=False)
  app.add_component(chatbot)
  app.run()

img 



**The Problem**: You have some NLP tasks that require manual data labeling and/or annotation.

**The Solution**: pykoi provides several predefined classes for different NLP-related annotation tasks:
- Summarization
- Question and Answering
- Named Entity Recognition
Simply create a dataset of annotation tasks, call them with the corresponding tasks, and serve 
them with pykoi. Users can access the shared url and complete the desired tasks. The data will
automatically be streamed to a database you can access, or you can just manually download the data
for yourself. pykoi even provides an additional layer of analytics on top of the annotated data,
so you can understand your newly-generated data.


### Summarization

### Question and Answering

### Named Entity Recognition
