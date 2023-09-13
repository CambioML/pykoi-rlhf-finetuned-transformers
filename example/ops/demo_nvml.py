"""Demo for the chatbot application."""
from pykoi import Application
from pykoi.component import Nvml

##############################################
# Nvidia Management Library (NVML) Component #
##############################################
ops_nvidia_gpu = Nvml()

########################################################
# Starting the application and add Nvml as a component #
########################################################
# Create the application
# app = pykoi.Application(debug=False, share=True)
app = Application(debug=False, share=False)
app.add_component(ops_nvidia_gpu)
app.run()
