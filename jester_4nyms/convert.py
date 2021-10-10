import numpy as np
import scipy.io
data = np.load('lam.npy')
np.savetxt('lam.csv', data, delimiter=',')
scipy.io.savemat('lam.mat', mdict={'lam': data})
#exit()
data = np.load('P.npy')
np.savetxt('P.csv', data, delimiter=',')
scipy.io.savemat('P.mat', mdict={'P': data})

data = np.load('Rvar.npy')
np.savetxt('Rvar.csv', data, delimiter=',')
scipy.io.savemat('Rvar.mat', mdict={'Rvar': data})

data = np.load('rows.npy')
np.savetxt('rows.csv', data, delimiter=',')
scipy.io.savemat('rows.mat', mdict={'rows': data})

data = np.load('columns.npy')
np.savetxt('columns.csv', data, delimiter=',')
scipy.io.savemat('columns.mat', mdict={'columns': data})

utilde=np.load('Utilde.npy')
v=np.load('v.npy')
rtilde = np.dot(utilde.T,v)
np.savetxt('rtilde.csv', rtilde, delimiter=',')
scipy.io.savemat('rtilde.mat', mdict={'rtilde': rtilde})
