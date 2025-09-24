import ollama
import re
from pyparsing import Word, alphas, Suppress, Group, Forward, OneOrMore, opAssoc, infixNotation
from pyparsing import Keyword, CaselessKeyword, printables
from pyparsing import restOfLine, quotedString
from collections import deque
from nltk.translate.bleu_score import sentence_bleu

from sympy import Symbol
from sympy.logic.boolalg import Equivalent

import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# Add a function to check for logical equivalence (conceptual)
def are_logically_equivalent(logic1: str, logic2: str, var_map: dict) -> bool:
    """
    CONCEPTUAL FUNCTION:
    Checks if two boolean logic strings are semantically equivalent.
    This would be implemented using a formal logic library like pyeda or sympy.logic.
    For now, it returns True only if the strings are identical, which is a weak check.
    """
    # In a real implementation:
    # 1. Parse logic1 and logic2 into ASTs.
    # 2. Normalize variables using var_map.
    # 3. Use pyeda.expr.expr('logic_string').equivalent(pyeda.expr.expr('other_logic_string'))
    # For this conceptual implementation, we'll just do a strict string comparison.
    return logic1.strip() == logic2.strip()


def measure_completeness(original_requirement: str, full_logic: str):
    """
    Measures LLM completeness by removing sentences and checking if the logic changes.
    """
    print("\n--- Measuring Completeness via Sentence Deletion ---")
    
    # 1. Split the original requirement into sentences
    sentences = sent_tokenize(original_requirement)
    if len(sentences) <= 1:
        print("  Skipping completeness check: Requirement is a single sentence.")
        return 0, 0 # Returns (sentences_incorporated, total_sentences)
    
    sentences_incorporated = 0
    total_sentences = len(sentences)
    
    for i, sentence_to_remove in enumerate(sentences):
        # 2. Create a modified requirement by removing one sentence
        modified_sentences = [s for j, s in enumerate(sentences) if i != j]
        modified_requirement = " ".join(modified_sentences)
        
        print(f"\n  --- Deleting Sentence #{i+1} ---")
        print(f"  Sentence Removed: '{sentence_to_remove}'")
        print(f"  Modified Requirement: '{modified_requirement}'")
        
        # 3. Get the LLM's new logic for the modified requirement
        try:
            modified_logic = ask_llm_for_boolean_logic(modified_requirement)
        except Exception as e:
            print(f"  Error getting modified logic from LLM: {e}")
            continue

        print(f"  Modified Logic: '{modified_logic}'")
        print(f"  Full Logic:     '{full_logic}'")

        # 4. Check for logical equivalence (Conceptual)
        # This part requires a real equivalence checker. For now, we'll just do a strict string match.
        is_equivalent = are_logically_equivalent(full_logic, modified_logic, {})
        
        if not is_equivalent:
            print("  -> LOGIC CHANGED. Sentence was successfully incorporated.")
            sentences_incorporated += 1
        else:
            print("  -> LOGIC DID NOT CHANGE. Sentence was NOT incorporated (possible incompleteness).")
            # Note: This could also mean the sentence was redundant.
            
    completeness_score = sentences_incorporated / total_sentences if total_sentences > 0 else 0
    print(f"\nCompleteness Score: {sentences_incorporated}/{total_sentences} ({completeness_score:.2f})")
    return sentences_incorporated, total_sentences

def ask_llm_for_boolean_logic(natural_language_requirement: str) -> str:
    prompt = f"""
    You are an expert in formal logic and system specifications.
    Your task is to translate natural language design requirements into Boolean logic expressions.

    Rules for Boolean Logic:
    - Use 'AND' for logical conjunction.
    - Use 'OR' for logical disjunction.
    - Use 'NOT' for logical negation.
    - Use 'IMPLIES' for logical implication (A IMPLIES B).
    - Use parentheses for grouping everywhere so that its neatly understandable.
    - Identify the clauses properly with more precision in the sentence and convert them into boolean specifications properly.
    - Variables should be capitalized single words (e.g., 'DOOR_OPEN', 'ALARM_ACTIVE').
    - Identify the main subject (e.g., 'software installation', 'system activation', 'alarm sounding') and include it as a variable in the final Boolean logic expression.
    - Do not include any explanations, preamble, or additional text. Only output the Boolean logic expression.
    - When you see 'only if' (not 'if') in the requirement like A only if B it is 'A implies B' not 'B implies A'
    - When you see 'if and only if' (not 'only if') in the requirement like A if and only if B it is 'A implies B AND B implies A'

    Natural Language Requirement: "{natural_language_requirement}"

    Boolean Logic:
    """
    try:
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1, 'num_predict': 128}
        )
        llm_output = response['message']['content'].strip()
        print(f"LLM Raw Output:\n{llm_output}\n")
        lines = [line.strip() for line in llm_output.split('\n') if line.strip()]
        if lines:
            return lines[0]
        else:
            return "Error: LLM returned empty response."
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

# --- Pyparsing Grammar and Helper Functions ---
def parse_boolean_logic(expression_string: str):
    AND = CaselessKeyword("AND")
    OR = CaselessKeyword("OR")
    NOT = CaselessKeyword("NOT")
    IMPLIES = CaselessKeyword("IMPLIES")
    variable_name = Word(alphas, alphas + "_")
    operand = Group(variable_name | Suppress("(") + Forward().setResultsName("nested_expr") + Suppress(")"))
    boolean_expression = Forward()
    boolean_expression <<= infixNotation(
    operand,
    [
        (NOT, 1, opAssoc.RIGHT),  # Highest precedence
        (AND, 2, opAssoc.LEFT),
        (OR, 2, opAssoc.LEFT),
        (IMPLIES, 2, opAssoc.RIGHT),  # Lowest precedence
    ]
    )
    try:
        parsed_expression = boolean_expression.parseString(expression_string, parseAll=True)
        return parsed_expression[0]
    except Exception as e:
        print(f"Error parsing expression: {e}")
        return None

def _get_operator_and_operands(node):
    if not isinstance(node, list):
        return None, node
    if len(node) == 2 and isinstance(node[0], str) and node[0].upper() == 'NOT':
        return node[0].upper(), [node[1]]
    if len(node) > 3 and all(isinstance(node[i], str) and node[i].upper() == node[1].upper() for i in range(1, len(node), 2)):
        current_op = node[1].upper()
        last_op_index = -1
        for i in range(len(node) - 1, 0, -1):
            if isinstance(node[i], str) and node[i].upper() == current_op:
                last_op_index = i
                break
        if last_op_index != -1:
            left_subtree_node = node[0:last_op_index]
            right_operand_node = node[last_op_index+1]
            return current_op, [left_subtree_node, right_operand_node]
    elif len(node) == 3 and isinstance(node[1], str) and node[1].upper() in ['AND', 'OR', 'IMPLIES']:
        return node[1].upper(), [node[0], node[2]]
    elif len(node) == 1:
        return _get_operator_and_operands(node[0])
    return None, None

def visualize_tree(node, level=0, prefix="Node: "):
    indent = "  " * level
    op, operands = _get_operator_and_operands(node)
    if op:
        print(f"{indent}{prefix}{op}")
        if op == 'NOT':
            visualize_tree(operands[0], level + 1, "Operand: ")
        elif op in ['AND', 'OR', 'IMPLIES']:
            visualize_tree(operands[0], level + 1, "Left: ")
            visualize_tree(operands[1], level + 1, "Right: ")
    elif operands is not None:
        print(f"{indent}{prefix}{operands}")
    else:
        print(f"{indent}{prefix}UNEXPECTED NODE (Please report): {node} (Type: {type(node).__name__})")
        if isinstance(node, list):
            for item in node:
                visualize_tree(item, level + 1, "Item: ")

def pretty_print_tree(node, indent=0):
    prefix = "  " * indent
    op, operands = _get_operator_and_operands(node)
    if op:
        print(f"{prefix}{op}")
        for operand in operands:
            pretty_print_tree(operand, indent + 1)
    elif operands is not None:
        print(f"{prefix}{operands}")
    else:
        print(f"{prefix}UNEXPECTED NODE (Please report): {node} (Type: {type(node).__name__})")
        if isinstance(node, list):
            for item in node:
                pretty_print_tree(item, indent + 1)

def extract_variables(node, variables: set):
    op, operands = _get_operator_and_operands(node)
    if op:
        for operand in operands:
            extract_variables(operand, variables)
    elif operands is not None:
        variables.add(operands)

# --- New Function for Semantic Equivalence Mapping ---
def get_semantic_equivalents(variable_name: str) -> list[str]:
    """
    Asks the LLM for a list of semantically equivalent terms for a given variable.
    """
    prompt = f"""
    Given the following concept expressed as a capitalized variable name, provide a comma-separated list of 
    semantically equivalent or synonymous terms. The terms should also be in capitalized, underscore-separated format.
    Do not include any explanations, preamble, or the original term itself.
    
    Example:
    'TEMPERATURE_HIGH' -> 'HIGH_TEMPERATURE,TEMP_EXCEEDS_THRESHOLD'
    'USER_AUTHENTICATED' -> 'AUTHENTICATED_USER,USER_LOGGED_IN'
    
    Input Variable: '{variable_name}'
    
    Equivalent Terms:
    """
    try:
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1, 'num_predict': 128}
        )
        llm_output = response['message']['content'].strip()
        terms = [term.strip() for term in llm_output.split(',') if term.strip()]
        return terms
    except Exception as e:
        print(f"Error getting semantic equivalents for '{variable_name}': {e}")
        return []

# --- Your original evaluation metrics (commented out for now) ---
def evaluate_metrics(llm_outputs: list[str], ground_truths: list[str]):
    """
    Evaluates a batch of LLM outputs against their corresponding ground truths.
    Calculates overall accuracy and false positive rate.
    """
    assert len(llm_outputs) == len(ground_truths), "Mismatched list lengths"
    R = W = 0
    GT = RS = len(ground_truths)
    for llm_output, ground_truth in zip(llm_outputs, ground_truths):
        llm_clean = llm_output.strip().replace(" ", "")
        gt_clean = ground_truth.strip().replace(" ", "")
        if llm_clean == gt_clean:
            R += 1
        else:
            W += 1
    accuracy = R / GT if GT > 0 else 0
    false_positive = W / RS if RS > 0 else 0
    return accuracy, false_positive, R, W, GT, RS

def calculate_bleu_score(llm_output: str, ground_truth: str) -> float:
    reference = [ground_truth.split()]
    candidate = llm_output.split()
    return sentence_bleu(reference, candidate)

# This function sends a prompt to the LLM to perform deconstruction.
def ask_llm_for_deconstruction(natural_language_requirement: str) -> str:
    prompt = f"""
    You are an expert in formal logic and system specifications.
    Your task is to break down a complex natural language requirement into a list of simple, atomic sentences.
    Each sentence should represent a single, clear logical proposition.

    Rules for Output:
    - Return a comma-separated list of the simple sentences.
    - Do not include any explanations, preamble, or additional text.
    - Do not use any boolean logic keywords (AND, OR, NOT, IMPLIES).
    
    Example:
    Input: "If a fire is detected, an alert must activate, and the system should only be active during business hours."
    Output: "a fire is detected, an alert must activate, the system should only be active during business hours"

    Natural Language Requirement: "{natural_language_requirement}"

    Simple Sentences:
    """
    try:
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1, 'num_predict': 256}
        )
        llm_output = response['message']['content'].strip()
        return llm_output
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

# This is the new `deconstruct_requirement` function.
def deconstruct_requirement(natural_language: str) -> list[str]:
    """
    Asks an LLM to deconstruct a single, complex requirement into a list of simple sentences.
    """
    print("DEBUG: Asking LLM to deconstruct the requirement.")
    llm_output_string = ask_llm_for_deconstruction(natural_language)
    
    # Parse the comma-separated list from the LLM output.
    simple_sentences = [s.strip() for s in llm_output_string.split(',') if s.strip()]
    
    return simple_sentences

def are_logically_equivalent_conceptual(logic1: str, logic2: str) -> bool:
    """
    CONCEPTUAL IMPLEMENTATION: Checks if two boolean logic strings are semantically equivalent
    using a formal logic library. This code is for demonstration and will not run
    without the 'sympy' library installed.
    """
    try:
        normalized_logic1 = logic1
        normalized_logic2 = logic2
        
        parsed_expr1 = parse_boolean_expression_to_sympy(normalized_logic1)
        parsed_expr2 = parse_boolean_expression_to_sympy(normalized_logic2)
        
        return bool(Equivalent(parsed_expr1, parsed_expr2))

    except Exception as e:
        print(f"Error during logical equivalence check: {e}")
        return False

def parse_boolean_expression_to_sympy(expression: str):

    expression = expression.replace('AND', '&').replace('OR', '|').replace('NOT', '~').replace('IMPLIES', '>>')
    variables = re.findall(r'\b[A-Z_]+\b', expression)
    symbols = {var: Symbol(var) for var in variables}

    return expression


def check_logic(c: str, s_list: list[str], d: str, pos: int, n: int, logic_main: str) -> None:
    if(pos==n-1):
        prompt_and_s = f"{d} Additionally, {s_list[pos]}."
        print(f"\n  Testing Another Case: ")
        print(f"  Input Prompt: '{prompt_and_s}'")
        
        logic_and_s = ask_llm_for_boolean_logic(prompt_and_s)
        print(f"  LLM Output: {logic_and_s}")
        s_boolean = ask_llm_for_boolean_logic(s_list[pos])
        curr_logic = f"{logic_main} AND ({s_boolean})"

        are_they = are_logically_equivalent_conceptual(logic_and_s, curr_logic)
        if(are_they):
            print("Consistent!!!!!!")
        else:
            print("Not Consistent!!!!!")

        print("\n------------------------------------")

        prompt_and_not_s = f"{d} However, it is not the case that {s_list[pos]}."
        print(f"\n  Testing Another Case: ")
        print(f"  Input Prompt: '{prompt_and_not_s}'")
        logic_and_not_s = ask_llm_for_boolean_logic(prompt_and_not_s)
        print(f"  LLM Output: {logic_and_not_s}")
        not_logic = f"It is not the case that {s_list[pos]}"
        s_boolean = ask_llm_for_boolean_logic(not_logic)
        curr_logic = f"{logic_main} AND ({s_boolean})"

        are_they = are_logically_equivalent_conceptual(logic_and_not_s, curr_logic)
        if(are_they):
            print("Consistent!!!!!!")
        else:
            print("Not Consistent!!!!!")

        print("\n------------------------------------")

        prompt = f"{d}"
        print(f"\n  Testing Another Case: ")
        print(f"  Input Prompt: '{prompt}'")
        logic = ask_llm_for_boolean_logic(prompt)
        print(f"  LLM Output: {logic}")

        are_they = are_logically_equivalent_conceptual(logic, logic_main)
        if(are_they):
            print("Consistent!!!!!!")
        else:
            print("Not Consistent!!!!!")

        print("\n------------------------------------")
    else:
        prompt_and_s = f"{d} Additionally, {s_list[pos]}."
        s_boolean = ask_llm_for_boolean_logic(s_list[pos])
        curr_logic = f"{logic_main} AND ({s_boolean})"
        check_logic(c,s_list,prompt_and_s,pos+1,n,curr_logic)

        prompt_and_not_s = f"{d} However, it is not the case that {s_list[pos]}."
        not_logic = f"It is not the case that {s_list[pos]}"
        s_boolean = ask_llm_for_boolean_logic(not_logic)
        curr_logic = f"{logic_main} AND ({s_boolean})"
        check_logic(c,s_list,prompt_and_not_s,pos+1,n,curr_logic)

        prompt = f"{d}"
        check_logic(c,s_list,prompt,pos+1,n,logic_main)

def evaluate_compositional_logic(c: str, s_list: list[str], logic: str) -> None:
    """
    Evaluates the LLM's ability to handle compositional logic (c AND s) and (c AND (NOT s)).
    """
    print("\n--- Evaluating Compositional Logic ---")

    n = len(s_list)
    check_logic(c,s_list,c,0,n,logic)
    
    # for i, s in enumerate(s_list):
    #     # Case 1: c AND s
    #     prompt_and_s = f"{c} Additionally, {s}."
    #     print(f"\n  Testing Case s{i+1}: c AND s")
    #     print(f"  Input Prompt: '{prompt_and_s}'")
    #     logic_and_s = ask_llm_for_boolean_logic(prompt_and_s)
    #     print(f"  LLM Output: {logic_and_s}")

    #     # Case 2: c AND (NOT s)
    #     # Note: The 'not' is applied to the natural language to test LLM's negation handling.
    #     prompt_and_not_s = f"{c} However, it is not the case that {s}."
    #     print(f"\n  Testing Case s{i+1}: c AND (NOT s)")
    #     print(f"  Input Prompt: '{prompt_and_not_s}'")
    #     logic_and_not_s = ask_llm_for_boolean_logic(prompt_and_not_s)
    #     print(f"  LLM Output: {logic_and_not_s}")
    
    # print("\n------------------------------------")

# --- Define TEST_REQUIREMENTS with Human Gold Standards ---
TEST_REQUIREMENTS = [
    {
        "natural_language": "The primary display will show status alerts, but only if the system is active and there are no critical errors.",
        "ground_truth": "NOT SOFTWARE_INSTALL IFF (OPERATING_SYSTEM_OUTDATED AND NOT_ENOUGH_DISK_SPACE)"
    },
    {
        "natural_language": "The primary display will show status alerts, but only if the system is active and there are critical errors.",
        "ground_truth": "NOT SOFTWARE_INSTALL IFF (OPERATING_SYSTEM_OUTDATED AND NOT_ENOUGH_DISK_SPACE)"
    },
    {
        "natural_language": "The primary display will show status alerts, but only if the system is not active and there are no critical errors.",
        "ground_truth": "NOT SOFTWARE_INSTALL IFF (OPERATING_SYSTEM_OUTDATED AND NOT_ENOUGH_DISK_SPACE)"
    },
    {
        "natural_language": "The primary display will show status alerts, but only if the system is not active or there are critical errors.",
        "ground_truth": "NOT SOFTWARE_INSTALL IFF (OPERATING_SYSTEM_OUTDATED AND NOT_ENOUGH_DISK_SPACE)"
    },
    {
        "natural_language": "The primary display will not show status alerts, but only if the system is active and there are no critical errors.",
        "ground_truth": "NOT SOFTWARE_INSTALL IFF (OPERATING_SYSTEM_OUTDATED AND NOT_ENOUGH_DISK_SPACE)"
    },
    {
        "natural_language": "The primary display will not show status alerts, but only if the system is active and there are critical errors.",
        "ground_truth": "NOT SOFTWARE_INSTALL IFF (OPERATING_SYSTEM_OUTDATED AND NOT_ENOUGH_DISK_SPACE)"
    },
    {
        "natural_language": "The primary display will not show status alerts, but only if the system is not active and there are no critical errors.",
        "ground_truth": "NOT SOFTWARE_INSTALL IFF (OPERATING_SYSTEM_OUTDATED AND NOT_ENOUGH_DISK_SPACE)"
    },
    {
        "natural_language": "The primary display will not show status alerts, but only if the system is not active or there are critical errors.",
        "ground_truth": "NOT SOFTWARE_INSTALL IFF (OPERATING_SYSTEM_OUTDATED AND NOT_ENOUGH_DISK_SPACE)"
    },
    {
        "natural_language": "The primary display will show status alerts, but only if the system is active.",
        "ground_truth": "NOT SOFTWARE_INSTALL IFF (OPERATING_SYSTEM_OUTDATED AND NOT_ENOUGH_DISK_SPACE)"
    },
    {
        "natural_language": "The primary display will show status alerts, but only if there are no critical errors.",
        "ground_truth": "NOT SOFTWARE_INSTALL IFF (OPERATING_SYSTEM_OUTDATED AND NOT_ENOUGH_DISK_SPACE)"
    },
]

if __name__ == "__main__":
    llm_outputs = []
    ground_truths = []
    
    for i, req_data in enumerate(TEST_REQUIREMENTS):
        req = req_data["natural_language"]
        ground_truth_req = req_data["ground_truth"]
        
        print(f"\n--- Requirement {i+1} ---")
        print(f"Natural Language: \"{req}\"")
        print(f"Ground Truth Boolean Logic: {ground_truth_req}\n")

        # simple_sentences = deconstruct_requirement(req)
        # for i in simple_sentences:
        #     print(i)
        
        llm_boolean_logic_output = ask_llm_for_boolean_logic(req)
        print(f"LLM-generated Boolean Logic: {llm_boolean_logic_output}")
        # continue
        # evaluate_compositional_logic(req, simple_sentences, llm_boolean_logic_output)
        
        llm_outputs.append(llm_boolean_logic_output)
        ground_truths.append(ground_truth_req)

        parsed_tree = parse_boolean_logic(llm_boolean_logic_output)
        
        if parsed_tree:
            print("--- Pyparsing ParseResults Object (Raw) ---")
            print(parsed_tree)
            print("\n--- Visualizing the Parse Tree (Simplified Method) ---")
            visualize_tree(parsed_tree.asList())
            print("\n--- Pretty Printing the Parse Tree (More Structured Method) ---")
            pretty_print_tree(parsed_tree.asList())
            # print(parsed_tree.asList())
            print("\n--- Transforming 'IMPLIES' to 'OR NOT' (Conceptual) ---")
            def transform_implies(node):
                op, operands = _get_operator_and_operands(node)
                # print(op, operands)
                if op == 'IMPLIES':
                    # print(f"Transforming 'IMPLIES' node: {node}")
                    antecedent = transform_implies(operands[0])
                    consequent = transform_implies(operands[1])
                    return [['NOT', antecedent], 'OR' , consequent]
                elif op:
                    transformed_operands = [transform_implies(o) for o in operands]
                    # print(transformed_operands, "sdgk")
                    if op == 'NOT':
                        return ['NOT', transformed_operands[0]]
                    elif op in ['AND', 'OR']:
                        return [[transformed_operands[0]], op, [transformed_operands[1]]]
                elif operands is not None:
                    return operands
                else:
                    print(f"WARNING: transform_implies encountered unhandled node: {node}")
                    return node
            transformed_tree = transform_implies(parsed_tree.asList())
            # print("MEOW")
            # print(parsed_tree.asList())
            # print("meowwww")
            # print(transformed_tree) 
            print("Transformed tree for 'IMPLIES' (Conceptual):")
            pretty_print_tree(transformed_tree)
            print("\n" + "="*70 + "\n")
            continue
            # print("\n" + "="*50 + "\n")
            
            print("\n--- Processing LLM Output ---")
            
            llm_variables = set()
            extract_variables(parsed_tree.asList(), llm_variables)
            print(f"Extracted Variables from LLM: {llm_variables}")
            
            semantic_map = {}
            for var in llm_variables:
                equivalents = get_semantic_equivalents(var)
                print(f" -> Found equivalents for '{var}': {equivalents}")
                semantic_map[var] = [var] + equivalents
            
            print("\n--- Final Semantic Mapping Dictionary ---")
            print(semantic_map)
            
            parsed_human_tree = parse_boolean_logic(ground_truth_req)
            human_variables = set()
            if parsed_human_tree:
                 extract_variables(parsed_human_tree.asList(), human_variables)
                 print(f"\nExtracted Variables from Human Gold Standard: {human_variables}")
            

            original_req_text = req_data["natural_language"]
            full_logic = llm_boolean_logic_output
            
            sentences_incorporated, total_sentences = measure_completeness(original_req_text, full_logic)
            print("\n" + "="*70 + "\n")
        else:
            print("Skipping processing due to parsing error.")
            print("\n" + "="*70 + "\n")
            
    # print("\n--- Batch Metrics ---")
    # if llm_outputs and ground_truths:
    #     avg_bleu = sum(calculate_bleu_score(llm, gt) for llm, gt in zip(llm_outputs, ground_truths)) / len(llm_outputs)
    #     print(f"Average BLEU Score across all tests: {avg_bleu:.4f}")
    #     accuracy, false_positive, R, W, GT, RS = evaluate_metrics(llm_outputs, ground_truths)
    #     print(f"Evaluation Metrics:")
    #     print(f"  Accuracy (ACC): {accuracy:.4f} (Right: {R}, Total: {GT})")