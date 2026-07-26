class CodeNode:
    def __init__(self, function_name):
        self.name = function_name
        self.called_functions = []

class CodeGraph:
    def __init__(self):
        # Dictionary to store function names mapped to their CodeNode objects
        self.nodes = {}

    def add_function(self, name):
        if name not in self.nodes:
            self.nodes[name] = CodeNode(name)

    def add_dependency(self, caller, callee):
        """Creates a directed edge from the caller function to the callee."""
        self.add_function(caller)
        self.add_function(callee)
        
        # BUG 1 IS ON THE NEXT LINE
        # Hint: Look closely at the attribute name being accessed.
        caller_node = self.node[caller] 
        
        caller_node.called_functions.append(callee)

    def print_graph(self):
        """Prints out the dependency tree of the codebase."""
        prin
