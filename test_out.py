import numpy as np
x = np.arange(10).astype(np.float)
# with open('test.txt', 'w') as f:
#     for i in x:
#         f.write("%d %s"% (i, "hello"))
np.savetxt('test.out', x, delimiter='\n')