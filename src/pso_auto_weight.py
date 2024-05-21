import numpy as np
import matplotlib.pyplot as plt
import helpers
import pandas as pd
import os
import subprocess
import time    



helpers.run_bash_cmnd("/home/awwalola/Codes/lsq/build/chimes_lsq fm_setup.in")
time.sleep(300)
def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    matrix = np.array([list(map(float, line.strip().split())) for line in lines], dtype=np.float32)

    return matrix
A = read_matrix_from_file("./A.0000.txt")
b = read_matrix_from_file("./b.txt")

def write_to_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        print("Content has been written to", file_path)
    except Exception as e:
        print("An error occurred:", e)


def save_list_to_file(path, file_name, data_list):
    # Combine path and file name to create the complete file path
    file_path = f"{path}/{file_name}"

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write each element of the list to a new line
        for item in data_list:
            file.write(str(item) + '\n')

def write_matrix_to_file(matrix, file_path):
    with open(file_path, 'w') as file:
        for row in matrix:
            # Convert each element in the row to a string and join them with spaces
            line = ' '.join(map(str, row))
            file.write(line + '\n')

def force_function_out(weights, A, b):
    weightedA = np.zeros((A.shape[0],A.shape[1]),dtype=float)
    weightedb = np.zeros((A.shape[0],),dtype=float)

    for i in range(A.shape[0]):     # Loop over rows (atom force components)
            for j in range(A.shape[1]): # Loop over cols (variables in fit)
                weightedA[i][j] = A[i][j]*weights[i]
                weightedb[i]    = b[i]   *weights[i]


                 # x, residuals, rank, s = np.linalg.lstsq(A,b, rcond=None)
    U, S, Vt = np.linalg.svd(weightedA, full_matrices=False)

    # Solve the linear system using SVD
    x = np.dot(Vt.T, np.dot(np.diag(1/S), np.dot(U.T, weightedb)))
    actual = weightedb
    predicted  = np.dot(weightedA, x)
    
    return weightedA , weightedb, x , predicted

def no_weight_function(A,b):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Solve the linear system using SVD
    x = np.dot(Vt.T, np.dot(np.diag(1/S), np.dot(U.T, b)))
    actual = b
    predicted  = np.dot(A, x)
  
    return actual, predicted

def force_function(weights, A, b):
    weightedA = np.zeros((A.shape[0],A.shape[1]),dtype=float)
    weightedb = np.zeros((A.shape[0],),dtype=float)

    for i in range(A.shape[0]):     # Loop over rows (atom force components)
            for j in range(A.shape[1]): # Loop over cols (variables in fit)
                weightedA[i][j] = A[i][j]*weights[i]
                weightedb[i]    = b[i]   *weights[i]


                 # x, residuals, rank, s = np.linalg.lstsq(A,b, rcond=None)
    U, S, Vt = np.linalg.svd(weightedA, full_matrices=False)

    # Solve the linear system using SVD
    x = np.dot(Vt.T, np.dot(np.diag(1/S), np.dot(U.T, weightedb)))
    actual = weightedb
    predicted  = np.dot(weightedA, x)
    
    return actual, predicted
size = A.shape[0]

def objective_function(weights):
    actual_DF_rdf = pd.read_csv('./rdf_actual_dft.csv', header='infer', sep =";")
    actual_rdf = actual_DF_rdf[actual_DF_rdf.columns[1]]
    weightedA = np.zeros((A.shape[0],A.shape[1]),dtype=float)
    weightedb = np.zeros((A.shape[0],),dtype=float)

    for i in range(A.shape[0]):     # Loop over rows (atom force components)
            for j in range(A.shape[1]): # Loop over cols (variables in fit)
                weightedA[i][j] = A[i][j]*weights[i]
                weightedb[i]    = b[i]   *weights[i]


                 # x, residuals, rank, s = np.linalg.lstsq(A,b, rcond=None)
    U, S, Vt = np.linalg.svd(weightedA, full_matrices=False)

    # Solve the linear system using SVD
    x = np.dot(Vt.T, np.dot(np.diag(1/S), np.dot(U.T, weightedb)))
    actual = weightedb
    predicted  = np.dot(weightedA, x)

    save_list_to_file("./", "Ax.txt", predicted)
    #save_list_to_file("./", "force.txt", predicted)
    save_list_to_file("./", "b.txt", actual)
    save_list_to_file("./", "x.txt", x)
    #write_matrix_to_file(A,"./A.txt")
    helpers.run_bash_cmnd("rm -f params.txt")
    #helpers.run_bash_cmnd_to_file("params.txt", "python3 /home/awwalola/Codes/lsq/build/chimes_lsq.py")
    helpers.run_bash_cmnd_to_file("params.txt", "python3 /home/awwalola/Codes/lsq/build/chimes_lsq.py --algorithm=dlasso --read_output True")
    command_run_md = "/home/awwalola/Codes/lsq/build/chimes_md run_md.in"
    helpers.writelines("run_md.out",helpers.run_bash_cmnd(command_run_md))
    time.sleep(3600)
    #bash_script_output = subprocess.check_output(['bash', 'word_count_script.sh'])
    #wc = bash_script_output.decode('utf-8').strip()
    #helpers.run_bash_cmnd("python3 /home/awwalola/Codes/al_driver-LLfork/src/dftbgen_to_xyz.py " +wc+ " traj.gen")
    bash_script_output = subprocess.check_output(['bash', 'your_script.sh'])
    bash_output_string = bash_script_output.decode('utf-8').strip()
    command_run_travis = "/nfs/turbo/coe-rklinds/software/travis-src-220729/exe/travis -p traj.xyz -i input.txt"
    helpers.run_bash_cmnd(command_run_travis)
    write_to_file("./name", bash_output_string)
    time.sleep(800)
    chimes_rdf_DF = pd.read_csv("./"+bash_output_string, header='infer', sep =";")
    time.sleep(200)
    chimes_rdf = chimes_rdf_DF[chimes_rdf_DF.columns[1]]
    save_list_to_file("./", "chimes_rdf.csv", chimes_rdf)
    save_list_to_file("./", "actual_rdf.csv", actual_rdf)
    rmse = np.sqrt(np.sum((actual_rdf - chimes_rdf) ** 2)/size)
    
    return rmse 

class Particle:
    def __init__(self, num_variables):
        self.position = np.random.rand(num_variables)
        self.velocity = np.random.rand(num_variables)
        self.best_position = self.position.copy()
        self.best_score = objective_function([*self.position])

def particle_swarm_optimization(num_particles, num_variables, max_iterations, inertia_weight, cognitive_weight, social_weight):
    particles = [Particle(num_variables) for _ in range(num_particles)]

    # Initialize global best with a valid position
    global_best_position = np.random.rand(num_variables)
    global_best_score = objective_function([*global_best_position])
    global_best_scores = []

    for iteration in range(max_iterations):
        for particle in particles:
            # Update velocity
            if global_best_position is not None:
                particle.velocity = (inertia_weight * particle.velocity +
                                    cognitive_weight * np.random.rand() * (
                                                particle.best_position - particle.position) +
                                    social_weight * np.random.rand() * (global_best_position - particle.position))

            # Update position
            particle.position += particle.velocity

            # Update personal best
            current_score = objective_function([*particle.position])
            if current_score.all() < particle.best_score.all():
                particle.best_position = particle.position.copy()
                particle.best_score = current_score

            # Update global best
            if current_score.all() < global_best_score.all():
                global_best_position = particle.position.copy()
                global_best_score = current_score
                
        global_best_scores.append(global_best_score) 
            

    return global_best_position, global_best_score, global_best_scores

if __name__ == "__main__":
    num_particles = 50
    num_variables = size
    max_iterations = 100
    inertia_weight = 0.8
    cognitive_weight = 1.5
    social_weight = 1.5


    best_position, best_score, global_best_scores = particle_swarm_optimization(num_particles, num_variables, max_iterations,
                                                             inertia_weight, cognitive_weight, social_weight)
    save_list_to_file("./", "weighted_list.txt", best_position)


