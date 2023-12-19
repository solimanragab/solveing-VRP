import numpy as np
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import ttk

# Helper Functions
def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Function to evaluate the total distance of the VRP solution
def eval_vrp_solution(solution, customer_locs, depot_loc, num_vehicles):
    """Evaluates the total distance of the VRP solution."""
    total_distance = 0.0
    for vehicle_tour in np.array_split(solution, num_vehicles):
        if len(vehicle_tour) == 0:
            continue
        vehicle_distance = distance(depot_loc, customer_locs[vehicle_tour[0]])
        for idx in range(1, len(vehicle_tour)):
            vehicle_distance += distance(customer_locs[vehicle_tour[idx-1]], customer_locs[vehicle_tour[idx]])
        vehicle_distance += distance(customer_locs[vehicle_tour[-1]], depot_loc)
        total_distance += vehicle_distance
    return total_distance

# Differential Evolution functions
def mutate(target_idx, population, F):
    """DE mutation operation."""
    indices = list(range(len(population)))
    indices.remove(target_idx)
    
    a_idx, b_idx, c_idx = random.sample(indices, 3)
    a, b, c = population[a_idx], population[b_idx], population[c_idx]
    
    # DE mutation operation
    mutant = [ai + F * (bi - ci) for ai, bi, ci in zip(a, b, c)]
    return [int(max(0, min(xi, len(customer_coords) - 1))) for xi in mutant]

def ensure_unique_customers(permutation, num_customers):
    """Ensures that each customer is included only once."""
    seen = set()
    unique_customers = [x for x in permutation if not (x in seen or seen.add(x))]
    remaining_customers = list(set(range(num_customers)) - set(unique_customers))
    random.shuffle(remaining_customers)
    return unique_customers + remaining_customers

def ensure_valid_permutation(individual, num_customers):
    """Correct the permutation to ensure that all customers are present exactly once."""
    if len(set(individual)) != len(individual):
        individual = ensure_unique_customers(individual, num_customers)
    return individual

def select_min_distance_individual(population, depot_loc, customer_locs, num_vehicles):
    """Selects the individual with the minimum route distance."""
    min_distance = float("inf")
    best_individual = None
    for individual in population:
        individual = ensure_valid_permutation(individual, len(customer_locs))
        distance = eval_vrp_solution(individual, customer_locs, depot_loc, num_vehicles)
        if distance < min_distance:
            min_distance = distance
            best_individual = individual
    return best_individual, min_distance

# Visualization of the solution
def plot_solution(depot_coords, customer_coords, solution, num_vehicles):
    plt.figure()
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, num_vehicles)))
    
    # Plot depot
    plt.scatter(*depot_coords, s=150, c='black', marker='*', label='Depot')
    
    for i, route_indices in enumerate(np.array_split(solution, num_vehicles)):
        color = next(colors)
        route_coords = [customer_coords[j] for j in route_indices]
        
        # Plot route
        plt.scatter(*zip(*route_coords), c=color, label=f'Vehicle {i+1}')
        plt.plot(*zip(*([depot_coords] + route_coords + [depot_coords])), c=color)
    
    plt.title('VRP Solution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()


# Differential Evolution main function
def differential_evolution_alg(customer_coords, depot_coords, num_customers, num_vehicles, pop_size, generations, F, CR):
    # Initialize the population
    pop = [np.random.permutation(num_customers) for _ in range(pop_size)]
    
    # Evolutionary process
    best_solution = None
    best_solution_fitness = np.inf
    for g in range(generations):
        for i in range(pop_size):
            # Mutation
            mutant = mutate(i, pop, F)
            # Recombination
            trial = crossover(mutant, pop[i], num_customers, CR)
            # Selection
            trial_fitness = eval_vrp_solution(trial, customer_coords, depot_coords, num_vehicles)
            if trial_fitness < eval_vrp_solution(pop[i], customer_coords, depot_coords, num_vehicles):
                pop[i] = trial

        # Track best solution
        current_best = min(pop, key=lambda x: eval_vrp_solution(x, customer_coords, depot_coords, num_vehicles))
        current_best_fitness = eval_vrp_solution(current_best, customer_coords, depot_coords, num_vehicles)
        if current_best_fitness < best_solution_fitness:
            best_solution_fitness = current_best_fitness
            best_solution = current_best
    
    return best_solution, best_solution_fitness

# Crossover Function:
def crossover(mutant, target, num_customers, CR):
    trial = []
    for i in range(num_customers):
        if random.random() < CR or i == random.randint(0, num_customers-1):
            trial.append(mutant[i])
        else:
            trial.append(target[i])
    return ensure_unique_customers(trial, num_customers)


# Main Differential Evolution Function
def create_input_fields():
    try:
        global num_customers
        num_customersInput = int(entry.get())  # Get the number of customers
        create_customer_input_fields(num_customersInput)
        num_customers = num_customersInput
    except ValueError:
        count_label.config(text="Please enter a valid number!")

def create_customer_input_fields(num_customersInput):
    global customer_entries, coordinate_labels, show_values_button, values_display_label, additional_input_frame, additional_entry, additional_button
    # Clear previous entries, labels, and show values button if they exist
    if 'customer_entries' in globals():
        for entry in customer_entries:
            entry.destroy()
    if 'coordinate_labels' in globals():
        for label in coordinate_labels:
            label.destroy()
    if 'show_values_button' in globals():
        show_values_button.destroy()
    if 'values_display_label' in globals():
        values_display_label.destroy()
    if 'additional_input_frame' in globals():
        additional_input_frame.destroy()

    # Create lists to hold Entry widgets and Labels for coordinates
    customer_entries = []
    coordinate_labels = []

    # Create input fields for each customer
    for i in range(1, num_customersInput + 1):
        label_text = f"Customer {i} coordinates (x, y):"
        customer_label = tk.Label(frame, text=label_text , font=("Arial", 12))
        customer_label.pack(pady=10)

        # Entry widgets for x and y coordinates
        x_entry = tk.Entry(frame , width=30 )
        y_entry = tk.Entry(frame , width=30)

        x_entry.pack(pady=2)
        y_entry.pack(pady=2)

        customer_entries.append((x_entry, y_entry))
        coordinate_labels.append(customer_label)

    # Show "Show Entered Values" button after creating input fields
    show_values_button = tk.Button(frame, text="Enter Values", command=show_values  , width=26)
    show_values_button.pack(pady=10)


    count_label.config(text="Please Enter The Required Data" , font=("Arial", 15))

def show_values():
    global customer_coords
    customer_coords = []
    values = []
    i = 0
    for x_entry, y_entry in customer_entries:
        x_val = x_entry.get()
        y_val = y_entry.get()
        i = i + 1
        customer_coords.append((float(x_val), float(y_val)))
        values.append(f"Customer {i} coordinates (x, y) : ( {float(x_val)} , {float(y_val)} )")

    if values:
        values_text = "\n\n".join(values)
        # Display entered values below the input fields
        global values_display_label
        values_display_label = tk.Label(frame, text=values_text)
        values_display_label.pack()

        # Create a new input field for the specified number
        global additional_input_frame, additional_entry, additional_button
        additional_input_frame = tk.Frame(frame)
        additional_input_frame.pack(pady=10)

        additional_entry = tk.Entry(additional_input_frame , width=30)
        additional_entry.grid(row=0, column=0, padx=5, pady=5)
        additional_entry.pack(side=tk.TOP, padx=5)

        additional_button = tk.Button(additional_input_frame, text="Enter Number Of Vehicles", command=perform_action , width=26)
        additional_button.pack(side=tk.BOTTOM, padx=5, pady=5 )
    else:
        count_label.config(text="No values entered yet!")

def perform_action():
    # Add your code here to perform a specific action based on the entered number
    global num_vehicles  # Declare as global to modify the global variable
    additional_number = additional_entry.get()
    num_vehicles = int(additional_number)
    # Create two new input fields for x and y after displaying values
    global additional_input_frame, additional_x_entry, additional_y_entry, additional_button

    label_text = f"Depot coordinates (x , y) :"
    customer_label = tk.Label(frame, text=label_text , font=("Arial", 15))
    customer_label.pack()

    additional_input_frame = tk.Frame(frame)
    additional_input_frame.pack(pady=10)

    additional_x_entry = tk.Entry(additional_input_frame , width=30)
    additional_y_entry = tk.Entry(additional_input_frame , width=30)

    additional_x_entry.pack(side=tk.TOP, pady=2)
    additional_y_entry.pack(side=tk.BOTTOM, pady=2)


    new_button = tk.Button(frame, text="Enter Depot Coordinates", command=add_value , width=26 )
    new_button.pack()


def add_value():
    # Add your code here to handle the added value
    global depot_coords  # Declare as global to modify the global variable
    x_val = additional_x_entry.get()
    y_val = additional_y_entry.get()
    depot_coords = (float(x_val), float(y_val))  # Convert to int and set depot_coords

    solve_button = tk.Button(frame, text="Solution" , command=solve , width=26 )
    solve_button.pack(padx=5, pady=15)

def solve():
    global customer_coords, depot_coords, num_customers, num_vehicles  # Declare as global to access these variables
    pop_size = 50  # Population size
    generations = 100  # Number of generations
    F = 0.8  # Differential weight
    CR = 0.9  # Crossover probability

    # Execute the DE algorithm
    best_solution, best_distance = differential_evolution_alg(
        customer_coords,
        depot_coords,
        num_customers,
        num_vehicles,
        pop_size,
        generations,
        F,
        CR
    )

    label_text = f"Best solution: {best_solution}"
    customer_label = tk.Label(frame, text=label_text  , font=("Arial", 18))
    customer_label.pack()

    label_text_distance = f"Best solution: {best_distance}"
    customer_label_distance = tk.Label(frame, text=label_text_distance , font=("Arial", 18))
    customer_label_distance.pack()

    plot_solution(depot_coords, customer_coords, best_solution, num_vehicles)


# Initialize Tkinter
root = tk.Tk()
root.title("Differential Algorithm")

# Create a Canvas widget
canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a scrollbar to the Canvas
scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the Canvas to use the scrollbar
canvas.configure(yscrollcommand=scrollbar.set)

# Create a frame inside the Canvas to hold the content
frame = tk.Frame(canvas)
frame.pack(pady=50)
canvas.create_window((0, 0), window=frame, anchor='n')

# Update the scroll region when the frame size changes
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame.bind("<Configure>", on_frame_configure)


entry = tk.Entry(frame , width=30)
entry.grid(row=0, column=0, padx=5, pady=5)
entry.pack(side=tk.TOP, padx=5)

count_button = tk.Button(frame, text="Enter number of customers: ", command=create_input_fields , width=26 )
count_button.pack(padx=5, pady=5 )

# Create a label to display status/information
count_label = tk.Label(frame, text="")
count_label.pack()

root.mainloop()