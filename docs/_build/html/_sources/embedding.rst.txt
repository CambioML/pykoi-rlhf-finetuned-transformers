Vista
===================================

**Vista** (/vi'sta/) is our tool to visualize embeddings in three dimensions:

img 

It can be used to understand question/answer, documents used in knowledge retrieval, to 
compare models to each other, and to annotate a set of data.

The python interface is easy to use, simply import the model and then call it on a set of:


.. code:: python

  import pykoi as pk
  # files
  files = pk.UploadFiles('./path/to/files')
  
  # visualize files with vista
  viz = pk.vista(files)

  # launch app
  app = pk.App()
  app.add_component(viz)
  app.run()

This will give you a sharable interface with the embedding model:
