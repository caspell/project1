import numpy as np

values = np.random.random((1, 10))

print ( values )

result = np.argmax(values)

print ( result )

print ( values[0,result] )