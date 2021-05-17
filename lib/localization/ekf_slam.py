import numpy as np

ROBOT_DIM = 3 # [x, y, th] in SE(2)
LANDMARK_DIM = 2 # [x, y] in R2
CONTROL_DIM = 2 # [dp, dth] in R2
OBSERV_DIM = 2 # [x, y] in R2
ASSOCIATION_DIM = 3 # [h, z, a_hz] in R3
NOISE_U = 0.1 # control noise constant part
NOISE_Z = 0.1 # measurement noise constant part
NOISE_L = 0.01 # new landmark initial noise
# yellow dashes are spaced 1 [inch] = 2,54 [cm]
GATING_TAU = 0.0254 # omega L2 distance gating tau
LONELY_GAMMA = 1e-4 # lonely best friend threshold
FRAME_RATE = 30 # [1/s]
DT = 1/FRAME_RATE # [s] env time interval
VERBOSE = True # full debug

class EKF_SLAM():
    def __init__(self):
        # initial state in SE2 [x, y, theta]
        self.mu = np.zeros((3,1), dtype=np.float32)
        self.sigma = np.zeros((3,3), dtype=np.float32) 
        print(f'mu={self.mu[:ROBOT_DIM].flatten()}')

    def step(self, controls: np.ndarray = np.zeros((CONTROL_DIM,),dtype=np.float32), \
                  observations: np.ndarray = np.zeros((1,LANDMARK_DIM),dtype=np.float32)):
        # incorporate new control
        self.predict(controls = controls)
        # incorporate new measurement
        self.correct(observations = observations)

    def predict(self, controls: np.ndarray = np.zeros((CONTROL_DIM,),dtype=np.float32)):
        # Incorporate the control by computing the
        # Gaussian distribution of the next state
        # given the input.
        # update the robot state
        self.transition(controls = controls)
        # state dimension
        state_dim = self.mu.shape[0]
        # robot state
        th = self.mu[2,0]
        # control controls
        # dp, dth are infinitesimal increment dp = DT * v , dth = DT * w
        dp = controls[0]
        # Linearize transition function around the current
        # state and measurement at each iteration. Derivatives 
        # are computed around the current mean of the estimate
        # Jacobian A: state transition matrix
        A = np.identity(state_dim,dtype=np.float32)
        A[:ROBOT_DIM,:ROBOT_DIM] = np.array([[1,0,-dp*np.sin(th)],\
                                            [0,1,dp*np.cos(th)],\
                                            [0,0,1]],dtype=np.float32)
        # Jacobian B: control input matrix
        B = np.zeros((state_dim,CONTROL_DIM),dtype=np.float32)
        B[:ROBOT_DIM,:CONTROL_DIM] = np.array([[np.cos(th), 0],\
                                                [np.sin(th),0],\
                                                [0,1]],dtype=np.float32)
        # inject noise on controls
        self.controls_noise(controls = controls, A = A, B = B)
        
    def transition(self, controls: np.ndarray = np.zeros((CONTROL_DIM,),dtype=np.float32)):
        # predict the robot motion, the transition model f(x,u)
        # only affects the robot pose not the landmarks
        # odometry euler integration
        x, y, th = self.mu[0,0], self.mu[1,0], self.mu[2,0]
        # dp, dth are infinitesimal increment dp = DT * v , dth = DT * w
        dp, dth = controls[0], controls[1]
        self.mu[0,0] = x + dp * np.cos(th)
        self.mu[1,0] = y + dp * np.sin(th)
        th = th + dth
        self.mu[2,0] = np.arctan2(np.sin(th),np.cos(th))

        if VERBOSE :
            print(f'mu={self.mu[:ROBOT_DIM].flatten()}')

    def controls_noise(self, controls: np.ndarray = np.zeros((CONTROL_DIM,),dtype=np.float32), \
                    A: np.ndarray = np.identity(ROBOT_DIM,dtype=np.float32), \
                    B: np.ndarray = np.identity(CONTROL_DIM,dtype=np.float32)):
        # Inject noise into model controls
        # control noise has a constant part which is sigma_u and some
        # velocity dependent parts
        sigma_v = controls[0]; # translational velocity dependent part
        sigma_w = controls[1]; # rotational velocity dependent part
        #compose control noise covariance sigma_u
        sigma_uu = np.array([[NOISE_U+sigma_v**2, 0],\
                            [0, NOISE_U+sigma_w**2]],dtype=np.float32)
        # The next state is an affine transform of the
        # past state and the controls
        # predict sigma
        self.sigma = A @ self.sigma @ A.T + B @ sigma_uu @ B.T

    def measure(self, landmark_position: np.ndarray = np.zeros((LANDMARK_DIM,1),dtype=np.float32), \
                    landmark_index: int = 0) -> (np.ndarray, np.ndarray):
        # compute the measurement function
        # for a distance based measurement
        state_dim = self.mu.shape[0]
        t, th = self.mu[0:2,0].reshape((2,1)), self.mu[2,0]
        landmark_position = landmark_position.reshape((LANDMARK_DIM,1))
        c, s = np.cos(th), np.sin(th)
        RT = np.array([[c, s],[-s, c]], dtype = np.float32) # transposed rotation matrix
        dRT = np.array([[-s, c],[-c, -s]], dtype = np.float32) # derivative of transposed rotation matrix
        # where I predict i will see that landmark
        dt = landmark_position - t
        # measurement_prediction
        h = RT @ dt
        # init Jacobian
        C = np.zeros((2, state_dim))
        # Jacobian w.r.t robot
        C[:2,:2] = -RT
        C[:2,2:3] = dRT @ dt
        # Jacobian w.r.t landmark
        C[:,landmark_index:landmark_index+2] = RT
        return h, C
    
    def measures_noise(self, C: np.ndarray = np.identity(OBSERV_DIM,dtype=np.float32)) -> np.ndarray:
        # model noise on measurements and return information matrix
        sigma_z = np.identity(2,dtype=np.float32) * NOISE_Z
        sigma_mn = C @ self.sigma @ C.T + sigma_z
        #compute information matrix
        omega_mn = np.linalg.inv(sigma_mn)
        return omega_mn

    def cost_matrix(self, landmarks: np.ndarray = np.zeros((1,LANDMARK_DIM),dtype=np.float32), \
                    observations: np.ndarray = np.zeros((1,OBSERV_DIM),dtype=np.float32)) -> np.ndarray:
        # cost matrix is A is (M x N)
        # get number of landmarks and new observations
        m_dim = observations.shape[0]
        n_dim = landmarks.shape[0]
        # build the association matrix with maximum distance values
        A = np.ones((m_dim, n_dim),dtype=np.float32)*1e3
        #now we have to populate the association matrix
        #for all landmarks
        for n in range(n_dim):
            # extract landmark
            landmark_position = landmarks[n,:].reshape((LANDMARK_DIM,1))
            landmark_index = n * LANDMARK_DIM + ROBOT_DIM
            # compute measurement function
            h, C = self.measure(landmark_position = landmark_position, landmark_index = landmark_index)
            # noise on measurements
            omega_mn = self.measures_noise(C = C)
            #for all measurements
            for m in range(m_dim):
                # obtain current measurement
                z = observations[m,:].reshape((OBSERV_DIM,1))
                # compute likelihood for this measurement and landmark
                # Omega L2 Norm
                A[m,n] = (z - h).T @ omega_mn @ (z - h)
        return A

    def gating(self,  A: np.ndarray = np.eye(1,2,dtype=np.float32)) -> (np.ndarray, np.ndarray):
        # Gating: ignore all
        # associations whose cost is
        # higher than a threshold
        m = np.arange(0,A.shape[0])
        n = np.argmin(A,axis=1)
        a_mn = A[m,n]
        # if observations don't pass the gating,
        # it means they are new ones, thus put the 
        # indices in a list
        indices_ko = np.where(a_mn >= GATING_TAU)[0]
        new_indices = np.take(m, indices_ko, axis=0)
        # if observations pass the gating,
        # save the associations
        indices_ok = np.where(a_mn < GATING_TAU)[0]
        m = np.take(m, indices_ok, axis=0)[...,None]
        n = np.take(n, indices_ok, axis=0)[...,None]
        a_mn = np.take(a_mn, indices_ok, axis=0)[...,None]
        associations = np.concatenate([m,n,a_mn],axis=-1)

        return associations, new_indices

    def best_friend(self, associations: np.ndarray = np.zeros((1,ASSOCIATION_DIM),dtype=np.float32), \
                    A: np.ndarray = np.identity(2,dtype=np.float32)) -> (np.ndarray, np.ndarray):
        # Best friends: an association should be the best (i.e. minimum) of
        # both row and column landmark in the state
        if associations.shape[0] == 0 : return np.array([]).reshape(0,ASSOCIATION_DIM), np.array([])
        # min by columns
        mm = np.argmin(A,axis=0)
        # min by rows
        m, n = associations[:,0].astype(int), associations[:,1].astype(int)
        # take those who are in n 
        mm = np.take(mm,n)
        # take those obs whose a_mn is minimum of rows and cols
        indices = np.where(m==mm)[0]
        associations = np.take(associations, indices, axis=0)
        # doubtful associations are the ones that are not best friend,
        # save unassigned measurements
        indices = np.where(m!=mm)[0]
        pruned = np.take(m, indices, axis=0)
        return associations, pruned

    def lonely_best_friend(self,associations: np.ndarray = np.zeros((1,ASSOCIATION_DIM),dtype=np.float32), \
                    A: np.ndarray = np.identity(2,dtype=np.float32)) -> (np.ndarray, np.ndarray):
        # Lonely best friends: one measurement should be
        # only assigned to one single landmark.
        if associations.shape[0] == 0 : return np.array([]).reshape(0,ASSOCIATION_DIM), np.array([])
        m, n, a_mn = associations[:,0].astype(int), associations[:,1].astype(int), associations[:,2]
        # exclude current minimum
        A[m,n] = np.inf
        # now take the second minimum
        a_min_row = np.take(np.min(A,axis=1),m)
        a_min_col = np.take(np.min(A,axis=0),n)
        # check weather current associations are in safe reigion w.r.t second minimum
        cond = np.logical_or(a_min_row - a_mn < LONELY_GAMMA, a_min_col - a_mn < LONELY_GAMMA)
        indices = np.where(np.logical_not(cond))[0]
        associations = np.take(associations, indices, axis=0)
        # save observations that didn't pass third gating
        indices = np.where(cond)[0]
        pruned = np.take(m, indices, axis=0)
        return associations, pruned
        
    def prune_heuristics(self, A: np.ndarray = np.identity(2,dtype=np.float32)) -> (np.ndarray, np.ndarray, np.ndarray):
        # Bad associations are EVIL. To avoid the above cases
        # we can use three heuristics
        # 1. Gating
        associations, new_indices = self.gating(A = A)
        # 2. Best friends
        associations, bf_pruned_indices = self.best_friend(associations = associations, A = A)
        # 3. Lonely Best Friend
        associations, lbf_pruned_indices = self.lonely_best_friend(associations = associations, A = A)
        # pruned associations are the ones that may lead to bad associations
        doubtful_indices = np.concatenate([bf_pruned_indices, lbf_pruned_indices],axis=0)

        return associations, new_indices, doubtful_indices

    def associate(self, landmarks: np.ndarray = np.zeros((1,LANDMARK_DIM),dtype=np.float32), \
                    observations: np.ndarray = np.zeros((1,OBSERV_DIM),dtype=np.float32)) -> (np.ndarray, np.ndarray, np.ndarray):
        # with a greedy algorithm compute Omega L2 Norm cost matrix
        A = self.cost_matrix(landmarks = landmarks, observations = observations)
        # use heuristics to avoid bad associations
        associations, new_indices, doubtful_indices = self.prune_heuristics(A = A)

        if VERBOSE :
            print(f'match={associations.shape[0]}, new={new_indices.shape[0]}, pruned={doubtful_indices.shape[0]}')
        
        return associations, new_indices, doubtful_indices

    def correct(self, observations: np.ndarray = np.zeros((1,OBSERV_DIM),dtype=np.float32)):
        # Here two cases arise, REOBSERVED landmark and NEW landmark.
        # For simplicity we can divide the problem: first analyze only the reobservations landmark
        # and work with them as in a localization procedure (of course, with full Jacobian now).
        # With this reobservations landmark we compute the correction/update of mean and covariance.
        # Then, after the correction is done, we can simply add the new landmark expanding the
        # mean and the covariance.
        landmarks = self.mu[ROBOT_DIM:,0]
        # if there are no observations exit
        if observations.shape[0] == 0 : return 
        # if there are no landmark in the state yet, fill the state with all observations
        if landmarks.shape[0] == 0 : 
            self.add_landmarks(new_indices = np.arange(observations.shape[0]), observations = observations)
            return
        landmarks = landmarks.reshape((-1,LANDMARK_DIM)) # reshape N x 2
        # compute cost matrix and associate landmarks
        associations, new_indices, _ = self.associate(landmarks = landmarks, observations = observations)
        # count how many observations of old landmarks we have
        num_known = associations.shape[0]
        if num_known == 0 :
            self.add_landmarks(new_indices = np.arange(observations.shape[0]), observations = observations)
            return
        zt = np.zeros((num_known * OBSERV_DIM, 1))
        ht = np.zeros((num_known * OBSERV_DIM, 1))
        Ct = np.zeros((num_known * OBSERV_DIM, self.mu.shape[0]))
        for i in range(num_known):
          m, n = associations[i,0].astype(int), associations[i,1].astype(int)
          # measured landmark position
          z = observations[m,:].reshape(OBSERV_DIM,1)
          landmark_index = n * LANDMARK_DIM + ROBOT_DIM
          # estimated landmark position
          landmark_position = self.mu[landmark_index: landmark_index + LANDMARK_DIM].reshape((LANDMARK_DIM,1))
          h, C = self.measure(landmark_position = landmark_position, landmark_index = landmark_index)
          h = h.reshape(OBSERV_DIM,1)
          # fill zt, ht, Ct
          j = i * OBSERV_DIM
          zt[j:j+OBSERV_DIM,:] = z
          ht[j:j+OBSERV_DIM,:] = h
          Ct[j:j+OBSERV_DIM,:] = C
        # observation noise
        sigma_z = np.identity(2*num_known,dtype=np.float32) * NOISE_Z
        # Kalman gain
        K = self.sigma @ Ct.T @ np.linalg.inv(Ct @ self.sigma @ Ct.T + sigma_z)
        # innovation
        ni = zt - ht
        self.mu = self.mu + K @ ni
        # update sigma
        self.sigma = (np.identity(self.mu.shape[0], dtype=np.float32) - K @ Ct) @ self.sigma

        if VERBOSE :
            print(f'total landmarks={self.mu[ROBOT_DIM:].shape[0]//2}')

        # add new landmarks to the state
        if new_indices.shape[0] == 0 : return
        self.add_landmarks(new_indices = new_indices, observations = observations)
    
    def add_landmarks(self, new_indices: np.ndarray = np.zeros((1,),dtype=np.int32), \
                        observations: np.ndarray = np.zeros((1,OBSERV_DIM),dtype=np.float32)):
        num_new = new_indices.shape[0]
        if num_new == 0 : return
        t, th = self.mu[0:2,0].reshape((2,1)), self.mu[2,0]
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s],[s, c]], dtype = np.float32) # rotation matrix 
        new_landmarks = observations[new_indices,:]
        # landmark w.r.t robot view
        # landmark position in the world
        landmark_world = R @ new_landmarks.T + t
        landmark_world = landmark_world.T.reshape(-1,1)
        # get the next free index in state vector
        last_landmark_index = self.mu.shape[0]
        # put landmark in state vector
        self.mu = np.concatenate([self.mu, landmark_world], axis=0)
        # add covariance for new landmarks
        # initial noise assigned to a new landmark
        # for simplicity we put a high value only in the diagonal.
        # A more deeper analysis on the initial noise should be made.
        sigma_landmark = np.identity(LANDMARK_DIM * num_new,dtype=np.float32)*NOISE_L
        # make place for new landmarks in covariance matrix
        self.sigma = np.concatenate([self.sigma,np.zeros((LANDMARK_DIM * num_new,self.sigma.shape[1]),dtype=np.float32)],axis=0)
        self.sigma = np.concatenate([self.sigma,np.zeros((self.sigma.shape[0],LANDMARK_DIM * num_new),dtype=np.float32)],axis=1)
        # set the new landmark covariance block
        self.sigma[last_landmark_index:last_landmark_index+LANDMARK_DIM * num_new, \
                last_landmark_index:last_landmark_index+LANDMARK_DIM * num_new] = sigma_landmark