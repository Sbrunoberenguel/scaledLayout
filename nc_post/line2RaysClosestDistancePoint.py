import numpy as np

def line2RaysClosestDistancePoint(L,Xi):

    l1 = L[0]
    l2 = L[1]
    l3 = L[2]
    lBar1 = L[3]
    lBar2 = L[4]
    lBar3 = L[5]

    cc_x0 = np.array([[ 
                                                                    -(l2*lBar3 - l3*lBar2)*(l2**2 + l3**2),
                     - lBar1*l2**2*l3 + 3*l1*lBar3*l2**2 - 2*l1*lBar2*l2*l3 - lBar1*l3**3 + l1*lBar3*l3**2,
                       lBar1*l2**3 - l1*lBar2*l2**2 + lBar1*l2*l3**2 + 2*l1*lBar3*l2*l3 - 3*l1*lBar2*l3**2,
                       lBar2*l1**2*l3 - 3*l2*lBar3*l1**2 + 2*l2*lBar1*l1*l3 + lBar2*l3**3 - l2*lBar3*l3**2,
l1**2*(2*l2*lBar2 - 2*l3*lBar3) - l1*(2*lBar1*l2**2 - 2*lBar1*l3**2) - 2*l2*l3**2*lBar2 + 2*l2**2*l3*lBar3,
                     - lBar3*l1**2*l2 + 3*l3*lBar2*l1**2 - 2*l3*lBar1*l1*l2 - lBar3*l2**3 + l3*lBar2*l2**2,
                                                                     (l1*lBar3 - l3*lBar1)*(l1**2 + l3**2),
                     - lBar2*l1**3 + l2*lBar1*l1**2 - lBar2*l1*l3**2 - 2*l2*lBar3*l1*l3 + 3*l2*lBar1*l3**2,
                       lBar3*l1**3 - l3*lBar1*l1**2 + lBar3*l1*l2**2 + 2*l3*lBar2*l1*l2 - 3*l3*lBar1*l2**2,
                                                                     -(l1*lBar2 - l2*lBar1)*(l1**2 + l2**2)]])

    cc_x1 =  np.array([[ 
                                                                                                            -(l2*lBar3 - l3*lBar2)**2,
                                                                                        (l2*lBar3 - l3*lBar2)*(2*l1*lBar3 - l3*lBar1),
                                                                                       -(2*l1*lBar2 - l2*lBar1)*(l2*lBar3 - l3*lBar2),
                                                                                                         -l1*l3*(l2*lBar3 - l3*lBar2),
                                                                                                          l1*l2*(l2*lBar3 - l3*lBar2),
                                                            - l1**2*lBar3**2 + lBar1*l1*l3*lBar3 - l3**2*lBar2**2 + l2*l3*lBar2*lBar3,
2*l1**2*lBar2*lBar3 - lBar1*l1*l2*lBar3 - lBar1*l1*l3*lBar2 - l2**2*lBar2*lBar3 + l2*l3*lBar2**2 + l2*l3*lBar3**2 - l3**2*lBar2*lBar3,
                                                                                                          l1*l3*(l2*lBar3 - l3*lBar2),
                                                                                        lBar3*l1**2*l3 + lBar3*l3**3 + l2*lBar2*l3**2,
                                                                - 2*lBar3*l1**2*l2 + lBar2*l1**2*l3 - lBar2*l2**2*l3 - lBar3*l2*l3**2,
                                                            - l1**2*lBar2**2 + lBar1*l1*l2*lBar2 - l2**2*lBar3**2 + l3*l2*lBar2*lBar3,
                                                                                                         -l1*l2*(l2*lBar3 - l3*lBar2),
                                                                  lBar3*l1**2*l2 - 2*lBar2*l1**2*l3 - lBar2*l2**2*l3 - lBar3*l2*l3**2,
                                                                                        lBar2*l1**2*l2 + lBar2*l2**3 + l3*lBar3*l2**2,
                                                                                                      -l3*lBar2*(l1*lBar3 - l3*lBar1),
                                            l1*(l3*lBar2**2 + l2*lBar2*lBar3 - l3*lBar3**2) + l3**2*lBar1*lBar3 - 2*l2*l3*lBar1*lBar2,
                                                                                        -l3*(lBar3*l1**2 + lBar3*l3**2 + l2*lBar2*l3),
                                                                                         l1*(lBar3*l1**2 + lBar3*l3**2 + l2*lBar2*l3),
                                        lBar1*l2**2*lBar2 - l1*l2*lBar2**2 + l1*l2*lBar3**2 - 2*l3*lBar1*l2*lBar3 + l1*l3*lBar2*lBar3,
                                                                    l1**2*(l2*lBar3 + l3*lBar2) + 2*l2**2*l3*lBar2 + 2*l2*l3**2*lBar3,
                                                                                        -l1*(lBar3*l1**2 + lBar3*l3**2 + l2*lBar2*l3),
                                                                                        -l1*(lBar2*l1**2 + lBar2*l2**2 + l3*lBar3*l2),
                                                                                                      -l2*lBar3*(l1*lBar2 - l2*lBar1),
                                                                                        -l2*(lBar2*l1**2 + lBar2*l2**2 + l3*lBar3*l2),
                                                                                         l1*(lBar2*l1**2 + lBar2*l2**2 + l3*lBar3*l2)]])
    
    cc_x2 =  np.array([[ 
                                                                                                  -l3*lBar1*(l2*lBar3 - l3*lBar2),
                                                        - l2**2*lBar3**2 + lBar2*l2*l3*lBar3 - l3**2*lBar1**2 + l1*l3*lBar1*lBar3,
                                    lBar2*l3**2*lBar3 + l2*l3*lBar1**2 - 2*l1*lBar2*l3*lBar1 - l2*l3*lBar3**2 + l1*l2*lBar1*lBar3,
                                                                                    -l3*(lBar3*l2**2 + lBar3*l3**2 + l1*lBar1*l3),
                                                                                     l2*(lBar3*l2**2 + lBar3*l3**2 + l1*lBar1*l3),
                                                                                    (l1*lBar3 - l3*lBar1)*(2*l2*lBar3 - l3*lBar2),
l1*(l3*lBar1**2 + l3*lBar3**2 - l2*lBar2*lBar3) - l1**2*lBar1*lBar3 + 2*l2**2*lBar1*lBar3 - l3**2*lBar1*lBar3 - l2*l3*lBar1*lBar2,
                                                                                    lBar3*l2**2*l3 + lBar3*l3**3 + l1*lBar1*l3**2,
                                                                                                      l2*l3*(l1*lBar3 - l3*lBar1),
                                                            - lBar1*l1**2*l3 - 2*lBar3*l1*l2**2 - lBar3*l1*l3**2 + lBar1*l2**2*l3,
                                    lBar2*l1**2*lBar1 - l2*l1*lBar1**2 + l2*l1*lBar3**2 - 2*l3*lBar2*l1*lBar3 + l2*l3*lBar1*lBar3,
                                                                                    -l2*(lBar3*l2**2 + lBar3*l3**2 + l1*lBar1*l3),
                                                            2*lBar1*l1**2*l3 + lBar3*l1*l2**2 + 2*lBar3*l1*l3**2 + lBar1*l2**2*l3,
                                                                                    -l2*(lBar1*l1**2 + l3*lBar3*l1 + lBar1*l2**2),
                                                                                                        -(l1*lBar3 - l3*lBar1)**2,
                                                                                    (l1*lBar2 - 2*l2*lBar1)*(l1*lBar3 - l3*lBar1),
                                                                                                     -l2*l3*(l1*lBar3 - l3*lBar1),
                                                                                                      l1*l2*(l1*lBar3 - l3*lBar1),
                                                        - l1**2*lBar3**2 + lBar2*l1*l2*lBar1 + l3*l1*lBar1*lBar3 - l2**2*lBar1**2,
                                                            - lBar1*l1**2*l3 + lBar3*l1*l2**2 - lBar3*l1*l3**2 - 2*lBar1*l2**2*l3,
                                                                                                     -l1*l2*(l1*lBar3 - l3*lBar1),
                                                                                    lBar1*l1**3 + l3*lBar3*l1**2 + lBar1*l1*l2**2,
                                                                                                   l1*lBar3*(l1*lBar2 - l2*lBar1),
                                                                                     l2*(lBar1*l1**2 + l3*lBar3*l1 + lBar1*l2**2),
                                                                                    -l1*(lBar1*l1**2 + l3*lBar3*l1 + lBar1*l2**2)]])

    cc_x3 =  np.array([[ 
                                                                                                         l2*lBar1*(l2*lBar3 - l3*lBar2),
                                          lBar3*l2**2*lBar2 + l3*l2*lBar1**2 - 2*l1*lBar3*l2*lBar1 - l3*l2*lBar2**2 + l1*l3*lBar1*lBar2,
                                                              - l2**2*lBar1**2 + lBar3*l2*l3*lBar2 + l1*l2*lBar1*lBar2 - l3**2*lBar2**2,
                                                                                           l3*(lBar2*l2**2 + l1*lBar1*l2 + lBar2*l3**2),
                                                                                          -l2*(lBar2*l2**2 + l1*lBar1*l2 + lBar2*l3**2),
                                          lBar3*l1**2*lBar1 - l3*l1*lBar1**2 + l3*l1*lBar2**2 - 2*l2*lBar3*l1*lBar2 + l2*l3*lBar1*lBar2,
- l1**2*lBar1*lBar2 + l1*l2*lBar1**2 + l1*l2*lBar2**2 - lBar3*l1*l3*lBar2 - l2**2*lBar1*lBar2 - lBar3*l2*l3*lBar1 + 2*l3**2*lBar1*lBar2,
                                                                                          -l3*(lBar2*l2**2 + l1*lBar1*l2 + lBar2*l3**2),
                                                                                          -l3*(lBar1*l1**2 + l2*lBar2*l1 + lBar1*l3**2),
                                                                  2*lBar1*l1**2*l2 + 2*lBar2*l1*l2**2 + lBar2*l1*l3**2 + lBar1*l2*l3**2,
                                                                                         -(l1*lBar2 - l2*lBar1)*(l2*lBar3 - 2*l3*lBar2),
                                                                                          lBar2*l2**3 + l1*lBar1*l2**2 + lBar2*l2*l3**2,
                                                                  - lBar1*l1**2*l2 - lBar2*l1*l2**2 - 2*lBar2*l1*l3**2 + lBar1*l2*l3**2,
                                                                                                            l2*l3*(l1*lBar2 - l2*lBar1),
                                                                                                         l1*lBar2*(l1*lBar3 - l3*lBar1),
                                                              - l1**2*lBar2**2 + lBar3*l1*l3*lBar1 + l2*l1*lBar1*lBar2 - l3**2*lBar1**2,
                                                                                           l3*(lBar1*l1**2 + l2*lBar2*l1 + lBar1*l3**2),
                                                                                          -l1*(lBar1*l1**2 + l2*lBar2*l1 + lBar1*l3**2),
                                                                                          (l1*lBar2 - l2*lBar1)*(l1*lBar3 - 2*l3*lBar1),
                                                                  - lBar1*l1**2*l2 - lBar2*l1*l2**2 + lBar2*l1*l3**2 - 2*lBar1*l2*l3**2,
                                                                                          lBar1*l1**3 + l2*lBar2*l1**2 + lBar1*l1*l3**2,
                                                                                                           -l1*l3*(l1*lBar2 - l2*lBar1),
                                                                                                              -(l1*lBar2 - l2*lBar1)**2,
                                                                                                           -l2*l3*(l1*lBar2 - l2*lBar1),
                                                                                                            l1*l3*(l1*lBar2 - l2*lBar1)]])

    m1 = Xi[0,:]
    m2 = Xi[1,:]
    m3 = Xi[2,:]
    mBar1 = Xi[3,:]
    mBar2 = Xi[4,:]
    mBar3 = Xi[5,:]


    x0Term =  np.array([
       m1**3,
    m1**2*m2,
    m1**2*m3,
    m1*m2**2,
    m1*m2*m3,
    m1*m3**2,
       m2**3,
    m2**2*m3,
    m2*m3**2,
       m3**3])
    
    
    x1Term =  np.array([
          m1**3,
       m1**2*m2,
       m1**2*m3,
    m1**2*mBar2,
    m1**2*mBar3,
       m1*m2**2,
       m1*m2*m3,
    m1*m2*mBar1,
    m1*m2*mBar2,
    m1*m2*mBar3,
       m1*m3**2,
    m1*m3*mBar1,
    m1*m3*mBar2,
    m1*m3*mBar3,
          m2**3,
       m2**2*m3,
    m2**2*mBar1,
    m2**2*mBar3,
       m2*m3**2,
    m2*m3*mBar1,
    m2*m3*mBar2,
    m2*m3*mBar3,
          m3**3,
    m3**2*mBar1,
    m3**2*mBar2])

    X = np.concatenate((
        np.dot(cc_x1,x1Term),
        np.dot(cc_x2,x1Term),
        np.dot(cc_x3,x1Term),
        np.dot(cc_x0,x0Term)),axis=0)
    return X[:3]/X[3,:]
