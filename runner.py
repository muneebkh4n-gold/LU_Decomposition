import subprocess
import csv

def run_command(command, env=None):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    return result.stdout, result.stderr

def save_to_csv(data, filename):
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data)

def run_serial(size, output_file):
    command = ["./serial", str(size), "1"]
    stdout, _ = run_command(command)
    time_taken = float(stdout.split()[0])
    save_to_csv(["Serial", "1", size, f"{time_taken}s"], output_file)

def run_openmp(size, threads, output_file):
    env = {"OMP_NUM_THREADS": str(threads)}
    command = ["./omp", str(size), str(threads), "0"]
    stdout, _ = run_command(command, env=env)
    time_taken = float(stdout.split()[0])
    save_to_csv(["OpenMP", str(threads), size, f"{time_taken}s"], output_file)

def run_mpi(size, processes, output_file):
    command = ["mpirun", "-np", str(processes), "./mpi", str(size)]
    stdout, _ = run_command(command)
    time_taken = float(stdout.split()[0])
    save_to_csv(["MPI", str(processes), size, f"{time_taken}s"], output_file)
    save_to_csv(["\n"], output_file)

def get_matrix_size():
    size = input("Enter matrix size: ")
    return int(size)

def get_threads():
    threads = input("Enter number of threads(OpenMP): ")
    return int(threads)

def get_processes():
    processes = input("Enter number of processes(MPI): ")
    return int(processes)

def main():
    output_file = "output.csv"
    size = get_matrix_size()
    threads = get_threads()
    processes = get_processes()

    # Run Serial
    run_serial(size, output_file)

    # Run OpenMP
    run_openmp(size, threads, output_file)

    # Run MPI
    run_mpi(size, processes, output_file)

if __name__ == "__main__":
    main()
