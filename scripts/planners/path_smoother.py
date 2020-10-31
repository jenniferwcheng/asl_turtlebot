import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    #convert path to numpy array because otherwise it's a pain!
    path = np.array(path)
    #create the time vector buy finding the distance from each point and dividing by the straight line velocity
    N = len(path)
    t = np.zeros(N)
    for i in range(1, N):
        #get the distance between the points
        distance = np.linalg.norm(path[i, :] - path[i-1, :])
        #calc the time based on distance and velocity
        t[i] = distance/V_des + t[i-1]
    t_smoothed = np.arange(t[0], t[-1], dt);
    print(t_smoothed.size)
    
    #interpolate over the given path 
    x_tck = scipy.interpolate.splrep(t, path[:,0], s=alpha)
    y_tck = scipy.interpolate.splrep(t, path[:,1], s=alpha)
    
    #allocate for the trajectory
    traj_smoothed = np.zeros([len(t_smoothed),7])
    
    #generate the states
    traj_smoothed[:,0] = scipy.interpolate.splev(t_smoothed, x_tck)
    traj_smoothed[:,1] = scipy.interpolate.splev(t_smoothed, y_tck)
    traj_smoothed[:,3] = scipy.interpolate.splev(t_smoothed, x_tck, der=1)
    traj_smoothed[:,4] = scipy.interpolate.splev(t_smoothed, y_tck, der=1)
    traj_smoothed[:,2] = np.arctan2(traj_smoothed[:,4], traj_smoothed[:,3])
    traj_smoothed[:,5] = scipy.interpolate.splev(t_smoothed, x_tck, der=2)
    traj_smoothed[:,6] = scipy.interpolate.splev(t_smoothed, y_tck, der=2) 
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
