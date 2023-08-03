Comparing Models
===================================

Comparing models is a difficult task. pykoi makes it easy by allowing one to 
directly compare the performance of multiple models to each other. 

If you have multiple language models that you'd like to compare to each other on
a set of prompts or via an interactive session, you can use `pk.Compare`.


Here is how you may run an interactive chat session with multiple models:

.. code:: python

  import pykoi as pk

  # assume you have some model, endpoint, or api
  model_1 = your_first_model
  model_2 = your_second_model
  model_3 = your_third_model
  model_4 = your_fourth_model

  model_array = [
    model_1,
    model_2,
    model_3,
    model_4
  ]

  # give interactive chat ui with feedback
  chatbot = pk.Compare(models=model_array)

  # Create sharable link to the application
  app = pk.Application(debug=False, share=False)
  app.add_component(chatbot)
  app.run()

This will give you a sharable UI that you can use to compare models to each other:

img 

**Comparison Visuals**

You can also view the feedback directly with the built in dashboard. Just make sure to launch it,
either in the same view as the chatbot UI:

.. code:: python

  ...
  # Create sharable link to the application
  app = pk.Application(debug=False, share=False)
  app.add_component(chatbot)
  app.run()

Or as a standalone:

.. code:: python

  ...
  # Create sharable link to the application
  app = pk.Application(debug=False, share=False)
  app.add_component(chatbot)
  app.run()

Either of which will yield:

img


**Comparing Preset Questions**

Given a set of pre-defined questions, you can feed them in to the models as follows:

.. code:: python

  ...
  # Create sharable link to the application
  app = pk.Application(debug=False, share=False)
  app.add_component(chatbot)
  app.run()


Note, that because the questions are pre-defined, you can launch the chat to give the feedback yourself,
or, you can just view the language output:

img


**The Solution**: pykoi offers a `pk.Compare` class that lets you rank up to 5 language models 
against each other on a set of pre-defined questions, or via an interactive session. pykoi provides
a UI to query the models, a dashboard with results, and the option to export the data for additional 
use.

Quick Code Example:
