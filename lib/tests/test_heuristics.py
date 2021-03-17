import numpy as np
from ..localization import EKF_SLAM
from ..localization import *
import matplotlib.pyplot as plt
np.random.seed(41296)

def test_gating_void():
    obs = 0
    land = 15 # land is never 0
    A = np.ones((obs,land))*GATING_TAU
    associations, new_indices = ekf.gating(A = A)
    assert associations.shape[0] == A.shape[0], 'void failed'
    assert associations.shape[0] == 0, '1 to 1 failed'
    assert new_indices.shape[0] == 0, 'there should not be new landmarks'

def test_gating_1_to_1():
    obs = 15
    land = 15
    A = np.ones((obs,land))*GATING_TAU
    m = np.arange(A.shape[0])
    n = np.arange(A.shape[1]) if obs < land else np.arange(A.shape[0])%A.shape[1]
    np.random.shuffle(n)
    n = n[:m.shape[0]]
    size = m.shape[0] if m.shape[0]>n.shape[0] else n.shape[0]
    a_mn = np.random.uniform(0,GATING_TAU-1e-6,(size,))
    A[m,n] = a_mn
    associations, new_indices = ekf.gating(A = A)
    assert associations.shape[0] == A.shape[0], '1 to 1 failed'
    assert new_indices.shape[0] == 0, 'there should not be new landmarks'
    assert np.intersect1d(associations[:,0], new_indices).shape[0] == 0, 'this must be void intersection'

def test_gating_1_to_rand():
    obs = 15
    land = 15
    A = np.ones((obs,land))*GATING_TAU
    m = np.arange(A.shape[0])
    n = np.arange(A.shape[1]) if obs < land else np.arange(A.shape[0])%A.shape[1]
    np.random.shuffle(n)
    n = n[:m.shape[0]]
    size = m.shape[0] if m.shape[0]>n.shape[0] else n.shape[0]
    gate = np.random.randint(0,2,(size,))*GATING_TAU
    a_mn = np.random.uniform(0,GATING_TAU-1e-6,(size,)) + gate
    A[m,n] = a_mn
    associations, new_indices = ekf.gating(A = A)
    assert associations.shape[0] + new_indices.shape[0] == A.shape[0], '1 to rand failed'
    gated = np.zeros((size,))
    gated[new_indices] = GATING_TAU
    assert new_indices.shape[0] == np.sum(gate>0), 'there should be new landmarks'
    assert np.all(gate == gated), 'there should be new landmarks'
    assert np.intersect1d(associations[:,0], new_indices).shape[0] == 0, 'this must be void intersection'

def test_best_friend_1_to_rand():
    obs = 15
    land = 5
    A = np.ones((obs,land))*GATING_TAU
    m = np.arange(A.shape[0])
    n = np.arange(A.shape[1]) if obs < land else np.arange(A.shape[0])%A.shape[1]
    np.random.shuffle(n)
    n = n[:m.shape[0]]
    size = m.shape[0] if m.shape[0]>n.shape[0] else n.shape[0]
    gate = np.random.randint(0,2,(size,))*GATING_TAU
    a_mn = np.random.uniform(0,GATING_TAU-1e-6,(size,)) + gate
    A[m,n] = a_mn
    associations, new_indices = ekf.gating(A = A)
    associations, doubtful = ekf.best_friend(associations = associations, A = A)
    assert np.intersect1d(associations[:,0], new_indices).shape[0] == 0, 'this must be void intersection'
    assert np.intersect1d(associations[:,0], doubtful).shape[0] == 0, 'this must be void intersection'
    assert np.intersect1d(new_indices, doubtful).shape[0] == 0, 'this must be void intersection'

def lonely_best_friend_1_to_rand():
    obs = 15
    land = 5
    A = np.ones((obs,land))*GATING_TAU
    m = np.arange(A.shape[0])
    n = np.arange(A.shape[1]) if obs < land else np.arange(A.shape[0])%A.shape[1]
    np.random.shuffle(n)
    n = n[:m.shape[0]]
    size = m.shape[0] if m.shape[0]>n.shape[0] else n.shape[0]
    gate = np.random.randint(0,2,(size,))*GATING_TAU
    a_mn = np.random.uniform(0,GATING_TAU-1e-6,(size,)) + gate
    A[m,n] = a_mn
    associations, new_indices = ekf.gating(A = A)
    associations, doubtful = ekf.best_friend(associations = associations, A = A)
    assert associations.shape[0] + new_indices.shape[0] + doubtful.shape[0] == A.shape[0], '1 to rand failed'
    assert np.intersect1d(associations[:,0], new_indices).shape[0] == 0, 'this must be void intersection'
    assert np.intersect1d(associations[:,0], doubtful).shape[0] == 0, 'this must be void intersection'
    assert np.intersect1d(new_indices, doubtful).shape[0] == 0, 'this must be void intersection'
    
def lonely_best_friend_1_to_rand():
    obs = 15
    land = 5
    A = np.ones((obs,land))*GATING_TAU
    m = np.arange(A.shape[0])
    n = np.arange(A.shape[1]) if obs < land else np.arange(A.shape[0])%A.shape[1]
    np.random.shuffle(n)
    n = n[:m.shape[0]]
    size = m.shape[0] if m.shape[0]>n.shape[0] else n.shape[0]
    gate = np.random.randint(0,2,(size,))*GATING_TAU
    a_mn = np.random.uniform(0,GATING_TAU-1e-6,(size,)) + gate
    A[m,n] = a_mn
    associations, new_indices = ekf.gating(A = A)
    associations, doubtful_bf = ekf.best_friend(associations = associations, A = A)
    associations, doubtful_lbf = ekf.lonely_best_friend(associations = associations, A = A)
    doubtful = np.concatenate([doubtful_bf, doubtful_lbf],axis=0)
    assert associations[:,0].shape[0] == len(list(set(associations[:,0]))), 'elements are repeated'
    assert associations[:,1].shape[0] == len(list(set(associations[:,1]))), 'elements are repeated'
    assert new_indices.shape[0] == len(list(set(new_indices))), 'elements are repeated'
    assert doubtful.shape[0] == len(list(set(doubtful))), 'elements are repeated'
    assert np.intersect1d(associations[:,0], new_indices).shape[0] == 0, 'this must be void intersection'
    assert np.intersect1d(associations[:,0], doubtful).shape[0] == 0, 'this must be void intersection'
    assert np.intersect1d(new_indices, doubtful).shape[0] == 0, 'this must be void intersection'

def lonely_best_friend_void():
    obs = 15
    land = 5
    A = np.ones((obs,land))*GATING_TAU
    associations, new_indices = ekf.gating(A = A)
    associations, doubtful_bf = ekf.best_friend(associations = associations, A = A)
    associations, doubtful_lbf = ekf.lonely_best_friend(associations = associations, A = A)
    doubtful = np.concatenate([doubtful_bf, doubtful_lbf],axis=0)
    assert associations.shape[0] + new_indices.shape[0] + doubtful.shape[0] == A.shape[0], '1 to rand failed'
    assert np.intersect1d(associations[:,0], new_indices).shape[0] == 0, 'this must be void intersection'
    assert np.intersect1d(associations[:,0], doubtful).shape[0] == 0, 'this must be void intersection'
    assert np.intersect1d(new_indices, doubtful).shape[0] == 0, 'this must be void intersection'

if __name__ == "__main__":
    global ekf
    ekf = EKF_SLAM()
    test_gating_void()
    test_gating_1_to_1()
    test_gating_1_to_rand()
    test_best_friend_1_to_rand()
    lonely_best_friend_1_to_rand()
    lonely_best_friend_void()
    print("Everything passed")
