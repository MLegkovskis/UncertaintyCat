# modules/model_utils.py

import ast
import astor

def generate_reduced_model_function(original_function, constant_values, influential_indices, problem, code_str):
    # Parse the code string into an AST
    tree = ast.parse(code_str)

    # Get names of influential and non-influential variables
    influential_vars = [problem['names'][i] for i in influential_indices]
    non_influential_vars = [name for name in problem['names'] if name not in influential_vars]

    # Create a mapping from variable names to their original indices
    var_original_indices = {var_name: idx for idx, var_name in enumerate(problem['names'])}

    # Create a mapping from original indices to new indices
    index_mapping = {}
    new_idx = 0
    for idx in range(len(problem['names'])):
        var_name = problem['names'][idx]
        if var_name in influential_vars:
            index_mapping[idx] = new_idx
            new_idx += 1

    # Extract the function_of_interest node
    class FunctionExtractor(ast.NodeVisitor):
        def __init__(self):
            self.function_node = None

        def visit_FunctionDef(self, node):
            if node.name == 'function_of_interest':
                self.function_node = node

    extractor = FunctionExtractor()
    extractor.visit(tree)
    if not extractor.function_node:
        raise ValueError("function_of_interest not found in the code.")

    # Get the name of the argument to function_of_interest
    function_arg_name = extractor.function_node.args.args[0].arg

    # Create a NodeTransformer to modify the function
    class FunctionTransformer(ast.NodeTransformer):
        def __init__(self, influential_vars, non_influential_vars, constant_values, var_original_indices, index_mapping, function_arg_name):
            self.influential_vars = influential_vars
            self.non_influential_vars = non_influential_vars
            self.constant_values = constant_values
            self.var_original_indices = var_original_indices
            self.index_mapping = index_mapping
            self.function_arg_name = function_arg_name

        def visit_FunctionDef(self, node):
            # Modify the arguments
            node.args.args = [ast.arg(arg='X')]

            # Create unpacking of X into influential variables
            unpacking_line = ast.parse(f"{', '.join(self.influential_vars)} = X").body[0]

            # Remove assignments to non-influential variables and the original unpacking line
            new_body = []
            for stmt in node.body:
                # Remove the original unpacking line
                if isinstance(stmt, ast.Assign):
                    if (isinstance(stmt.targets[0], (ast.Tuple, ast.List)) and
                        isinstance(stmt.value, ast.Name) and stmt.value.id == self.function_arg_name):
                        continue  # Skip the original unpacking line
                    elif len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                        var_name = stmt.targets[0].id
                        if var_name in self.non_influential_vars:
                            # Skip assignments to non-influential variables
                            continue
                        elif var_name in self.influential_vars:
                            # Remove redundant assignments from X (since we've unpacked X)
                            if isinstance(stmt.value, ast.Subscript):
                                if (isinstance(stmt.value.value, ast.Name) and stmt.value.value.id == self.function_arg_name):
                                    continue
                new_body.append(stmt)

            # Insert constant assignments for non-influential variables
            constant_assignments = []
            for var in self.non_influential_vars:
                value = self.constant_values[var]
                assign_stmt = ast.Assign(
                    targets=[ast.Name(id=var, ctx=ast.Store())],
                    value=ast.Constant(value=value)
                )
                constant_assignments.append(assign_stmt)

            # Insert unpacking and constant assignments at the beginning
            node.body = [unpacking_line] + constant_assignments + new_body

            # Replace variable usage in the function body
            class VariableReplacer(ast.NodeTransformer):
                def __init__(self, function_arg_name, var_original_indices, index_mapping):
                    self.function_arg_name = function_arg_name
                    self.var_original_indices = var_original_indices
                    self.index_mapping = index_mapping

                def visit_FunctionDef(self, node):
                    # Do not modify nested functions
                    return node

                def visit_Name(self, node):
                    # Replace uses of the old function argument name with 'X'
                    if node.id == self.function_arg_name:
                        node.id = 'X'
                    return self.generic_visit(node)

                def visit_Subscript(self, node):
                    # Replace indices in X
                    if isinstance(node.value, ast.Name) and node.value.id == self.function_arg_name:
                        if isinstance(node.slice.value, ast.Constant):
                            original_idx = node.slice.value.value
                            if original_idx in index_mapping:
                                new_idx = index_mapping[original_idx]
                                node.slice.value.value = new_idx
                            else:
                                # Replace with the variable name (fixed variable)
                                var_name = problem['names'][original_idx]
                                return ast.copy_location(ast.Name(id=var_name, ctx=ast.Load()), node)
                    return self.generic_visit(node)

            replacer = VariableReplacer(self.function_arg_name, var_original_indices, index_mapping)
            node.body = [replacer.visit(stmt) for stmt in node.body]

            return node

    transformer = FunctionTransformer(
        influential_vars,
        non_influential_vars,
        constant_values,
        var_original_indices,
        index_mapping,
        function_arg_name
    )
    new_function_node = transformer.visit(extractor.function_node)

    # Replace the function in the tree
    class FunctionReplacer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'function_of_interest':
                return new_function_node
            else:
                return node

    tree = FunctionReplacer().visit(tree)

    # Update the problem definition
    class ProblemUpdater(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'problem'):
                new_problem = {
                    'num_vars': len(influential_vars),
                    'names': influential_vars,
                    'distributions': [problem['distributions'][problem['names'].index(var)] for var in influential_vars]
                }
                node.value = ast.parse(repr(new_problem)).body[0].value
                return node
            else:
                return node

    tree = ProblemUpdater().visit(tree)

    # Convert the AST back to code
    new_code = astor.to_source(tree)

    # Optional: Format the code using black for better readability
    try:
        import black
        mode = black.Mode()
        formatted_code = black.format_str(new_code, mode=mode)
        return formatted_code
    except ImportError:
        # If black is not installed, return unformatted code
        return new_code
