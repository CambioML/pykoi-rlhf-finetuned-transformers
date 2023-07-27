Finetuning & Knowledge Retrieval
===================================

The `pk.Knowledge` class makes it easy to finetune a model to a specific set of documents. 
You can do this via a file-upload API, or a glob of local paths.

**Finetune With File Upload UI**

To finetune an existing model via a file-upload UI, you can run the following:

.. code:: python

  import pykoi as pk

  # assume you have some model, endpoint, or api
  model_1 = your_model

  # give interactive chat ui with feedback
  chatbot = pk.Compare(model=model)
  fileupload = pk.FileUpload(model=model)

  # Create sharable link to the application
  app = pk.Application(debug=False, share=False)
  app.add_component(chatbot)
  app.add_component(fileupload)
  app.run()

Which will launch:

img 


This is a useful option when you want to provide users with models that they can finetune for themselves.


**Finetune via Code**

In addition, you can finetune your model on local documents and save it for later use.
Alternatively, you can finetune a model by specifying a set of local files:

.. code:: python

  import pykoi as pk

  # assume you have some model, endpoint, or api
  model_1 = your_model

  # give interactive chat ui with feedback
  chatbot = pk.Compare(model=model)
  fileupload = pk.FileUpload(model=model)

  # Create sharable link to the application
  app = pk.Application(debug=False, share=False)
  app.add_component(chatbot)
  app.add_component(fileupload)
  app.run()


Finetuned models can be saved for further use elsewhere, or shared via  pykoi's chat UI's:

Let's see a quick example where we allow one to upload their own documents in one page, and 
interact with the new model in another:

**Visualizing Your Finetuned Model**

To understand your documents, you may use the 



Local Files:
