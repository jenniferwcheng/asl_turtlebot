import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    
    #placeholder vars
    x, y, theta = xvec
    V, om = u
    dtheta = om*dt
    '''
    print(om)
    print(dt)
    print(dtheta)
    '''
    
    if abs(om) < EPSILON_OMEGA: # if omega is ~constant
        # calculate new states
        g = xvec + dt*np.array([V*np.cos(theta), V*np.sin(theta), om])
        
        #              dx dy dtheta
        Gx = np.array([[1.0, 0.0, -dt*V*np.sin(theta)],
                       [0.0, 1.0, dt*V*np.cos(theta)],
                       [0.0, 0.0, 1.0]])
        
        #                  dV           domega      
        Gu = np.array([[dt*np.cos(theta), -0.5*V*np.sin(theta)*dt**2],
                       [dt*np.sin(theta), 0.5*V*np.cos(theta)*dt**2],
                       [0.0, dt]]) 
        #print(Gu)
    else:# omega changes with time
        # calculate new states
        #g = xvec + dtheta*np.array([(V/om)*np.cos(theta), (V/om)*np.sin(theta), 1.0])
        g = xvec + np.array([(V/om)*(np.sin(theta+dtheta)-np.sin(theta)),(V/om)*(-np.cos(theta+dtheta)+np.cos(theta)),dtheta])
        
        #              dx dy dtheta
        Gx = np.array([[1.0, 0.0, (V/om)*(np.cos(theta+dtheta)-np.cos(theta))],
                       [0.0, 1.0, (V/om)*(np.sin(theta+dtheta)-np.sin(theta))],
                       [0.0, 0.0, 1.0]])
        
        #                       dV                                         
        Gu = np.array([[(1.0/om)*(np.sin(theta+dtheta)-np.sin(theta)), -(V/om**2)*(np.sin(theta+dtheta) - np.sin(theta)) + (V/om)*(np.cos(theta+dtheta)*dt)],
                       [(1.0/om)*(-np.cos(theta+dtheta)+np.cos(theta)), -(V/om**2)*(-np.cos(theta+dtheta) + np.cos(theta)) + (V/om)*(np.sin(theta+dtheta)*dt)], 
                       [0.0, dt]])

    ########## Code ends here ##########

    if not compute_jacobians:
        return g
    
    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line
    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)

    # rotate and translate camera from base to world frame 
    theta_r = x[2]
    xt, yt, thetat = tf_base_to_camera
    R = np.array([[np.cos(theta_r), -np.sin(theta_r), 0.0],
                  [np.sin(theta_r), np.cos(theta_r), 0.0],
                  [0.0, 0.0, 1.0]])
     
    #R*t               
    x_cam_world = x + np.dot(R,tf_base_to_camera)                       
                                       
    #break up into components to make things easier
    x_cam, y_cam, theta_cam = x_cam_world

    #calculate h
    h = np.zeros(2)
    #h = np.array([(alpha - theta_cam), (r - rho*np.cos(phi))]) 
    h[0] = alpha - theta_cam
    #h[1] = r - rho*np.cos(phi)
    #expand above equation through triangles to get r' = r - x_cam*cos(alpha) - y_cam*sin(alpha) 
    #makes Jacobian easier too
    h[1] = r-x_cam*np.cos(alpha) - y_cam*np.sin(alpha) 
    
    #create Jacobian
    #calculate dr/dx base
    drdxb = -np.cos(alpha)
    #calculate dr/dy base
    drdyb = -np.sin(alpha)
    #calculate dr/dtheta base NOTE:use expanded rotation eq
    drdthb = -np.cos(alpha)*(-xt*np.sin(theta_r) - yt*np.cos(theta_r)) - np.sin(alpha)*(xt*np.cos(theta_r) - yt*np.sin(theta_r))
    #need to differentiate wrt to thetab
    #constuct Jacobian matrix
    
    Hx = np.array([[0.0, 0.0, -1.0],[drdxb, drdyb, drdthb]])
   
   
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
