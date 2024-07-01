import numpy as np
import matplotlib.pyplot as plt

D = 5.13        # Diameter of gage circle
E = 110 * 10 ** 9  # Young's modulus
v = 0.32      # Poisson's ratio

#

# Constants from ASTM - E837 Table 5
ca1 = -0.57051
ca2 = -12.3313
ca3 = 101.6656
ca4 = -308.9886
ca5 = 353.8417

cb1 = -1.13026
cb2 = -19.4443
cb3 = 128.086 
cb4 = -299.832
cb5 = 240.4794

# Calculates calibration constants using Eq 16
depth = 0.1
a_bar = []
b_bar = []

while depth <= 1:
    h = depth
    a = ca1*(h / D) + ca2*((h / D)**2) + ca3*((h / D)**3) + ca4*((h / D)**4) + ca5*((h / D)**5) 
    b = cb1*(h / D) + cb2*((h / D)**2) + cb3*((h / D)**3) + cb4*((h / D)**4) + cb5*((h / D)**5) 
    depth_increments = 0.1
    depth += depth_increments
    a_bar.append(a)
    b_bar.append(b)
    
a_values = np.array(a_bar)
b_values = np.array(b_bar)

# Stress computation function
def thick_uniform_stress(filename):
    
    data = np.loadtxt(filename, delimiter='\t', skiprows=25)
    
    time = data[:, 0]
    e1 = data[:, 1]
    e2 = data[:, 2]
    e3 = data[:, 3]
    num_rows = len(e1)  
    
    # Set threshold values and window sizes
    gradient_threshold = 10
    skip_steps = 200
    window_size = 200
    window_back = 50
    
    # Lists to store average strains for each direction
    average_strains_e1 = []
    average_strains_e2 = []
    average_strains_e3 = []
    
    start_time = []
    start_e1 = []
    end_time = []
    end_e1 = []
    
    # Loop through the stress data
    i = 0
    while i < len(e1) - window_size:
        # Calculate gradients
        gradients = np.gradient(e1[i:i + window_size])
        
        # Find the first point where the gradient exceeds the threshold
        exceed_indices = np.where(np.abs(gradients) >= gradient_threshold)[0]
        if exceed_indices.size > 0:
            exceed_index = exceed_indices[0] + i
            
            # Go back by window_back steps and set the start and end of the window
            end_index = exceed_index - window_back
            start_index = max(0, end_index - window_size)
            
            if end_index <= len(e1):
                average_strain_e1 = np.mean(e1[start_index:end_index])
                average_strains_e1.append(average_strain_e1)
                average_strain_e2 = np.mean(e2[start_index:end_index])
                average_strains_e2.append(average_strain_e2)
                average_strain_e3 = np.mean(e3[start_index:end_index])
                average_strains_e3.append(average_strain_e3)
                
                # To plot points
                start_time.append(start_index / 50)
                start_e1.append(e1[start_index + 25]) # +25 for visibility
                
                end_time.append(end_index / 50)
                end_e1.append(e1[end_index + 25])
                
                # Skip forward by skip_steps
                i = exceed_index + skip_steps
            else:
                break
        else:
            i += 1
    
    # Append the last window size amount of points
    start_index = len(e1) - window_size
    end_index = len(e1)
        
    average_strain_e1 = np.mean(e1[start_index:end_index])
    average_strains_e1.append(average_strain_e1)
    average_strain_e2 = np.mean(e2[start_index:end_index])
    average_strains_e2.append(average_strain_e2)
    average_strain_e3 = np.mean(e3[start_index:end_index])
    average_strains_e3.append(average_strain_e3)
        
    # To plot points
    start_time.append(start_index / 50)
    start_e1.append(e1[start_index + 25]) # +25 for visibility
                
    end_time.append(end_index / 50)
    end_e1.append(e1[end_index - 1])
    
    # Convert average strain lists to NumPy arrays and scale them
    strains_e1 = (np.array(average_strains_e1)) * 10**-6 #This is strain A
    strains_e2 = (np.array(average_strains_e2)) * 10**-6 #This is strain C
    strains_e3 = (np.array(average_strains_e3)) * 10**-6 #This is strain B
    
    #Assigning variables
    
    thr = 3**(1/2)
    strain_xx = np.zeros(len(strains_e1))
    strain_xy = np.zeros(len(strains_e2))
    strain_yy = np.zeros(len(strains_e3))

    
    #Converting strain from delta rosette to expected values
    for i in range(len(strains_e2)):
        strain_xx[i] = strains_e1[i]  
        strain_yy[i] = (2*(strains_e3[i]+strains_e2[i])-strains_e1[i])/3 
        strain_xy[i] = (strains_e3[i]-strains_e2[i])/(-thr) 
        
    # Compute combination strains using Eq 3-5 
    p = (strain_yy[1:] + strain_xx[1:]) / 2
    q = (strain_yy[1:] - strain_xx[1:]) / 2
    t = (strain_yy[1:] + strain_xx[1:] - (2 * strain_xy[1:])) / 2  
    
    # Compute combination stresses using Eq 17-19
    # Computes to units of MPa
    
    if len(p) == len(a_values): 
        P = (-E / (1 + v)) * (np.sum(a_values * p) / np.sum(a_values**2)) * 10**-6
        Q = -E * (np.sum(b_values * q) / np.sum(b_values**2)) * 10**-6
        T = -E * (np.sum(b_values * t) / np.sum(b_values**2)) * 10**-6   
        
        # Compute in-plane Cartesian stresses Eq 11-13
        sigma_x = P - Q
        sigma_y = P + Q
        t_xy = T    
        
        # Calculate angle beta
        beta = 1/2*np.degrees(np.arctan2(-T, -Q))
        
        # Compute the principal stresses Eq 14
        principal1 = P + np.sqrt((Q**2) + (T**2))
        principal2 = P - np.sqrt((Q**2) + (T**2))  
        
        max_stress = min(principal1, principal2) #min, not max as negative FYI
        min_stress = max(principal1, principal2)
        
        print('')
        print(f'Min Principal stress = {np.round(max_stress, decimals=1)} MPa')
        print(f'Max Principal stress = {np.round(min_stress, decimals=1)} MPa')
        print(f'Angle = {np.round(beta, decimals=2)}')
        print(f'T = {np.round(T, decimals=2)} MPa')
        print(f'Q = {np.round(Q, decimals=2)} MPa')
        
        print('')
        print(f'Sigma x = {np.round(sigma_x, decimals=1)} MPa')
        print(f'sigma y = {np.round(sigma_y, decimals=1)} MPa')  
        print(f't_xy = {np.round(t_xy, decimals=3)} MPa')  
        
    else:
        print(len(p))
        print(len(a_values))   

    plt.plot(time, e1, color='red')
    plt.plot(time, e2, color='blue')
    plt.plot(time, e3, color='black')
    #plt.plot(start_time, start_e1, 'bo')
    #plt.plot(end_time, end_e1, 'ro')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain (Micro strain)')
    plt.title("Pre HT Turbine 2, Site 2")
    plt.legend(['SG1', 'SG2', 'SG3'])
    plt.show()
    
fig, ax = plt.subplots(figsize=(9.825, 5), dpi=500)

thick_uniform_stress("954_anneal1.txt")
