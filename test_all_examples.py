import os
import sys
import time
import traceback
import numpy as np
import openturns as ot
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}" + "="*80)
    print(f" {text}")
    print("="*80 + f"{Style.RESET_ALL}")

def print_success(text):
    """Print a success message."""
    print(f"{Fore.GREEN}{Style.BRIGHT}✓ {text}{Style.RESET_ALL}")

def print_error(text):
    """Print an error message."""
    print(f"{Fore.RED}{Style.BRIGHT}✗ {text}{Style.RESET_ALL}")

def print_info(text):
    """Print an info message."""
    print(f"{Fore.YELLOW}{Style.BRIGHT}ℹ {text}{Style.RESET_ALL}")

def test_monte_carlo(model, problem, n_samples=100):
    """
    Run a simple Monte Carlo simulation to test if the model works.
    
    Args:
        model: OpenTURNS model function
        problem: OpenTURNS joint distribution
        n_samples: Number of samples to generate
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Generate random samples from the input distribution
        input_sample = problem.getSample(n_samples)
        
        # Evaluate the model on the input samples
        output_sample = model(input_sample)
        
        # Check if there are any NaN or inf values in the output
        if np.isnan(output_sample).any() or np.isinf(output_sample).any():
            print_error("Monte Carlo simulation produced NaN or inf values")
            return False
        
        # Calculate basic statistics
        mean = output_sample.computeMean()
        std_dev = output_sample.computeStandardDeviation()
        
        print_info(f"Mean: {mean[0]:.6g}, Std Dev: {std_dev[0]:.6g}")
        return True
    except Exception as e:
        print_error(f"Monte Carlo simulation failed: {str(e)}")
        return False

def test_example(file_path):
    """
    Test a single example file.
    
    Args:
        file_path: Path to the example file
        
    Returns:
        bool: True if all tests passed, False otherwise
    """
    file_name = os.path.basename(file_path)
    print_header(f"Testing {file_name}")
    
    try:
        # Create a fresh namespace
        namespace = {}
        
        # Read the file content
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Execute the code in the namespace
        exec(code, namespace)
        
        # Check if model and problem are defined
        if 'model' not in namespace:
            print_error("'model' is not defined in the example")
            return False
        
        if 'problem' not in namespace:
            print_error("'problem' is not defined in the example")
            return False
        
        model = namespace['model']
        problem = namespace['problem']
        
        # Check model input and output dimensions
        input_dim = model.getInputDimension()
        output_dim = model.getOutputDimension()
        print_info(f"Model dimensions: Input={input_dim}, Output={output_dim}")
        
        # Check problem dimension
        problem_dim = problem.getDimension()
        print_info(f"Problem dimension: {problem_dim}")
        
        # Verify dimensions match
        if input_dim != problem_dim:
            print_error(f"Dimension mismatch: model input ({input_dim}) != problem ({problem_dim})")
            return False
        
        # Test sampling from the distribution
        print_info("Testing distribution sampling...")
        try:
            test_sample = problem.getSample(10)
            print_success("Successfully sampled from the distribution")
        except Exception as e:
            print_error(f"Failed to sample from distribution: {str(e)}")
            return False
        
        # Test model evaluation
        print_info("Testing model evaluation...")
        try:
            test_input = problem.getSample(1)
            test_output = model(test_input)
            print_success("Successfully evaluated the model")
        except Exception as e:
            print_error(f"Failed to evaluate the model: {str(e)}")
            return False
        
        # Run Monte Carlo simulation
        print_info("Running Monte Carlo simulation...")
        mc_success = test_monte_carlo(model, problem)
        if mc_success:
            print_success("Monte Carlo simulation completed successfully")
        
        # Overall success
        if mc_success:
            print_success(f"All tests passed for {file_name}")
            return True
        else:
            print_error(f"Some tests failed for {file_name}")
            return False
            
    except Exception as e:
        print_error(f"Error testing {file_name}: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main function to test all examples."""
    examples_dir = 'examples'
    
    # Get all Python files in the examples directory
    example_files = [f for f in os.listdir(examples_dir) if f.endswith('.py')]
    example_files.sort()  # Sort alphabetically
    
    print_header(f"Found {len(example_files)} example files to test")
    
    # Track results
    successful = []
    failed = []
    
    # Test each example
    for i, file_name in enumerate(example_files):
        file_path = os.path.join(examples_dir, file_name)
        print(f"\n[{i+1}/{len(example_files)}] ", end="")
        
        start_time = time.time()
        success = test_example(file_path)
        elapsed_time = time.time() - start_time
        
        if success:
            successful.append(file_name)
            print_info(f"Completed in {elapsed_time:.2f} seconds")
        else:
            failed.append(file_name)
            print_info(f"Failed after {elapsed_time:.2f} seconds")
    
    # Print summary
    print_header("SUMMARY")
    print(f"Total examples: {len(example_files)}")
    print(f"{Fore.GREEN}Successful: {len(successful)}{Style.RESET_ALL}")
    print(f"{Fore.RED}Failed: {len(failed)}{Style.RESET_ALL}")
    
    if failed:
        print_error("The following examples failed:")
        for file_name in failed:
            print(f"  - {file_name}")
    else:
        print_success("All examples passed!")
    
    # Return exit code based on success
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
