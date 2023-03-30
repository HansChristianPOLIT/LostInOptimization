"""BufferStockModel

Solves the Deaton-Carroll buffer-stock one-asset consumption model with either:
A. vfi: standard value function iteration - start with this
B. nvfi: nested value function iteration
C. egm: endogenous grid point method

"""


############
# 2. model #
############

class OneAssetModelClass(ModelClass):
