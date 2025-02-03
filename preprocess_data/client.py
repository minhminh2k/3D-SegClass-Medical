import synapseclient
import synapseutils 
syn = synapseclient.Synapse() 

syn.login(authToken="Synapse_Token")

path = "./data/Synapse"
files = synapseutils.syncFromSynapse(syn, 'syn3193805', path=path) 

import os
