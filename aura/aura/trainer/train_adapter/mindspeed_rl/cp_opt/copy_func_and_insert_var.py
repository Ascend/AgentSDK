import ast
import os

# Python 3.9+ 可用 ast.unparse，否则需安装 astor
try:
    from ast import unparse
except ImportError:
    try:
        import astor

        def unparse(node):
            return astor.to_source(node).rstrip()
    except ImportError:
        raise ImportError("Please install astor: pip install astor")

from mindspeed_rl.utils.loggers import Loggers

# Initialize logger
logger = Loggers('copy_func_and_insert_var')


def copy_func_and_insert_var(
    src_file: str, dst_file: str, func_name: str, var_name: str = None, var_value: str = "{}"
) -> bool:
    """
    Replace a function in the target file with the one from the source file, and optionally insert/update a global variable.

    This function extracts the specified function from the source file (using AST for standard formatting),
    replaces it in the target file (preserving indentation context), and optionally inserts or updates
    a global variable at the correct position (after all global imports, before the first function/class).

    Args:
        src_file (str): Path to the source Python file.
        dst_file (str): Path to the target Python file.
        func_name (str): Name of the function to replace.
        var_name (str, optional): Name of the global variable to insert/update. If None, no variable is inserted.
        var_value (str, optional): Value of the global variable. Defaults to "{}".

    Returns:
        bool: True if operation succeeded.

    Raises:
        FileNotFoundError: If source or target file does not exist.
        ValueError: If the function is not found in source or target file.
    """
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Source file does not exist: {src_file}")
    if not os.path.exists(dst_file):
        raise FileNotFoundError(f"Target file does not exist: {dst_file}")

    # === 1. Extract function node from source file ===
    with open(src_file, 'r', encoding='utf-8') as f:
        src_code = f.read()

    src_tree = ast.parse(src_code)
    src_func_node = None

    for node in ast.walk(src_tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            src_func_node = node
            break

    if not src_func_node:
        raise ValueError(f"Function '{func_name}' not found in source file: {src_file}")

    # Generate standard format function source code (no comments, no extra whitespace)
    try:
        func_source = unparse(src_func_node)
    except Exception as e:
        raise ValueError(f"Failed to unparse function '{func_name}' from source file: {src_file}. Error: {e}") from e

    # === 2. Process target file ===
    with open(dst_file, 'r', encoding='utf-8') as f:
        dst_code = f.read()

    dst_tree = ast.parse(dst_code)

    # Use NodeTransformer to locate the target function
    class FunctionReplacer(ast.NodeTransformer):
        def __init__(self, target_name, new_func_source):
            self.target_name = target_name
            self.new_func_source = new_func_source
            self.found = False

        def visit_FunctionDef(self, node):
            if node.name == self.target_name:
                self.found = True
                return node  # Do not modify AST; we will replace text later
            return self.generic_visit(node)

    replacer = FunctionReplacer(func_name, func_source)
    new_dst_tree = replacer.visit(dst_tree)

    if not replacer.found:
        raise ValueError(f"Function '{func_name}' not found in target file: {dst_file}")

    # === 3. Text replacement: adjust indentation to match target context ===
    dst_lines = dst_code.splitlines(keepends=True)

    # Locate target function position (line numbers)
    target_node = None
    for node in ast.walk(dst_tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target_node = node
            break

    if not target_node:
        raise ValueError(f"Function '{func_name}' not found in target file during line lookup")

    start_line = target_node.lineno - 1
    end_line = getattr(target_node, 'end_lineno', target_node.lineno)

    # Get indentation level of target function definition line
    target_def_line = dst_lines[start_line]
    target_indent = len(target_def_line) - len(target_def_line.lstrip())

    # Adjust function source indentation to match target context
    func_lines = func_source.splitlines()
    if not func_lines:
        return False

    # First line is function definition
    first_line = func_lines[0]
    first_indent = 0  # AST-generated function definition has 0 indent

    # Calculate relative indentation difference for each line
    relative_indents = []
    for line in func_lines:
        current_indent = len(line) - len(line.lstrip())
        relative_indent = current_indent - first_indent
        relative_indents.append(relative_indent)

    # Rebuild each line with adjusted indentation
    adjusted_lines = []
    for i, line in enumerate(func_lines):
        relative_indent = relative_indents[i]
        new_indent = target_indent + relative_indent
        adjusted_line = ' ' * max(0, new_indent) + line.lstrip()
        adjusted_lines.append(adjusted_line)

    adjusted_func_source = '\n'.join(adjusted_lines) + '\n'

    # Replace text
    dst_lines[start_line:end_line] = [adjusted_func_source]

    new_code = ''.join(dst_lines)

    # === 4. (Optional) Insert/update global variable ===
    if var_name is not None:
        lines = new_code.splitlines(keepends=True)

        # Build variable definition string
        var_def = f"{var_name} = {var_value}"

        # Remove all old variable definitions (simple regex matching)
        new_lines = []
        for line in lines:
            if line.strip().startswith(var_name + ' ='):
                continue
            new_lines.append(line)

        # Find all global-scope import statements (indentation = 0)
        global_import_indices = []
        for i, line in enumerate(new_lines):
            stripped = line.strip()
            if (stripped.startswith('import ') or stripped.startswith('from ')) and len(line) - len(line.lstrip()) == 0:
                global_import_indices.append(i)

        # Find last global import index
        last_global_import_index = global_import_indices[-1] if global_import_indices else -1

        # Find first function/class definition (indentation = 0)
        first_def_index = -1
        for i, line in enumerate(new_lines):
            stripped = line.strip()
            if (stripped.startswith('def ') or stripped.startswith('class ')) and len(line) - len(line.lstrip()) == 0:
                first_def_index = i
                break

        # Determine insertion position
        if last_global_import_index != -1:
            insert_pos = last_global_import_index + 1
        else:
            insert_pos = 0

        # If there's a function/class definition after insert_pos, insert before it
        if first_def_index != -1 and first_def_index > insert_pos:
            insert_pos = first_def_index

        # Ensure blank line before insertion (optional, for readability)
        if insert_pos > 0 and new_lines[insert_pos - 1].strip() != '':
            new_lines.insert(insert_pos, '\n')
            insert_pos += 1

        # Insert new variable
        new_lines.insert(insert_pos, '\n' + var_def + '\n')
        new_code = ''.join(new_lines)

    # === 5. Write back to target file ===
    with open(dst_file, 'w', encoding='utf-8') as f:
        f.write(new_code)

    if var_name is not None:
        logger.info(
            f"Successfully replaced function '{func_name}' and inserted/updated global variable '{var_name}' in {dst_file}"
        )
    else:
        logger.info(f"Successfully replaced function '{func_name}' in {dst_file}")

    return True


# ================== Example Usage ==================
if __name__ == "__main__":
    """
    Example:
    python aura/trainer/train_adapter/mindspeed_rl/cp_opt/copy_func_and_insert_var.py aura/trainer/train_adapter/mindspeed_rl/cp_opt/utils.py third_party/rl/mindspeed/mindspeed/core/context_parallel/utils.py get_selection_indices_for_tnd_softmax_update _SOFTMAX_INDICES_CACHE_LRU "{}"
    python aura/trainer/train_adapter/mindspeed_rl/cp_opt/copy_func_and_insert_var.py aura/trainer/train_adapter/mindspeed_rl/cp_opt/utils.py third_party/rl/mindspeed/mindspeed/core/context_parallel/utils.py accumulate_list _ACCUMULATE_LIST_CACHE_LRU "{}"
    python aura/trainer/train_adapter/mindspeed_rl/cp_opt/copy_func_and_insert_var.py aura/trainer/train_adapter/mindspeed_rl/cp_opt/dot_product_attention.py third_party/rl/mindspeed_llm/mindspeed_llm/core/transformer/dot_product_attention.py do_ring_context_parallel
    """
    import sys

    if len(sys.argv) < 4:
        logger.error("Usage:")
        logger.error(
            "  Replace only function: python copy_func_and_insert_var.py <source.py> <target.py> <function_name>"
        )
        logger.error(
            "  Replace function + insert variable: python copy_func_and_insert_var.py <source.py> <target.py> <function_name> <var_name> [var_value]"
        )
        logger.error("Example:")
        logger.error("  python copy_func_and_insert_var.py A.py B.py forward_update")
        logger.error("  python copy_func_and_insert_var.py A.py B.py forward_update _SOFTMAX_INDICES_CACHE_LRU \"{}\"")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2]
    func = sys.argv[3]

    # Check if variable parameters are provided
    if len(sys.argv) >= 5:
        var_name = sys.argv[4]
        var_value = sys.argv[5] if len(sys.argv) > 5 else "{}"
    else:
        var_name = None
        var_value = "{}"

    try:
        success = copy_func_and_insert_var(src, dst, func, var_name, var_value)
        if success:
            logger.info("✅ Operation completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)
