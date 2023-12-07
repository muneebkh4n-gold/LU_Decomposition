import subprocess
import csv

def run_command(command, env=None):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    return result.stdout, result.stderr

def save_to_csv(data, filename):
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data)

def run_serial(size, output_file, processed_combinations):
    combination = ("Serial", "1", size)
    if combination not in processed_combinations:
        command = ["./serial", str(size), "1"]
        stdout, _ = run_command(command)
        time_taken = float(stdout.split()[0])
        save_to_csv(["Serial", "1", size, f"{time_taken}s"], output_file)
        processed_combinations.add(combination)

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

def get_threads():
    threads = input("Enter number of threads(OpenMP): ")
    return int(threads)

def get_processes():
    processes = input("Enter number of processes(MPI): ")
    return int(processes)

def main():
    output_file = "output.csv"
    matrix_sizes = [1000, 5000, 10000]
    thread_combinations = [2, 4, 8, 16]
    processed_combinations = set()

    for size in matrix_sizes:
        for threads in thread_combinations:
            # Run Serial only if the combination hasn't been processed
            run_serial(size, output_file, processed_combinations)

            # Run OpenMP
            run_openmp(size, threads, output_file)

            # Run MPI
            # Assuming the number of processes is the same as the number of threads for simplicity
            run_mpi(size, threads, output_file)

        print()  # Add a line break after each iteration

if __name__ == "__main__":
    main()
