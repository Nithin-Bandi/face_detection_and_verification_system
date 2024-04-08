import pandas as pd
import numpy as np

image_shape = (224, 224, 3)
encodingArr = np.empty((0,) + image_shape)
labels=np.empty((0))
np.save('utilsFiles\\encodingArr.npy',encodingArr)
np.save('utilsFiles\\labelArr.npy',labels)

