"""Demo for the chatbot application."""
import pykoi

##############################################
# Nvidia Management Library (NVML) Component #
##############################################
ops_nvidia_gpu = pykoi.Nvml()

########################################################
# Starting the application and add Nvml as a component #
########################################################
# Create the application
# app = pykoi.Application(debug=False, share=True)
app = pykoi.Application(debug=False, share=False)
app.add_component(ops_nvidia_gpu)
app.run()
