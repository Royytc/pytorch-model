import numpy as np
a=np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
print(a.shape)
a1=a[:,:,:,np.newaxis]
print(a1.shape)
print(a1)
a2=a[np.newaxis,:,:,:]
print(a2.shape)
print(a2)
a3=a[:,:,np.newaxis,:]
print(a3.shape)
print(a3)
a4=a[:,np.newaxis,:,:]
print(a4.shape)
print(a4)