import ollama
import re
from pyparsing import Word, alphas, Suppress, Group, Forward, OneOrMore, opAssoc, infixNotation
from pyparsing import Keyword, CaselessKeyword, printables
from pyparsing import restOfLine, quotedString
from collections import deque
from nltk.translate.bleu_score import sentence_bleu

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
    - Variables should be capitalized single words (e.g., 'DOOR_OPEN', 'ALARM_ACTIVE').
    - Do not include any explanations, preamble, or additional text. Only output the Boolean logic expression.
    - And when you see only if in the requirement like A only if B it is 'A implies B' not 'B implies A'
    - give parantheses for every single operation the boolean logic but dont give it for the entire logic

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
    boolean_expression <<= infixNotation(operand,
        [
            (IMPLIES, 2, opAssoc.RIGHT),
            (OR, 2, opAssoc.LEFT),
            (AND, 2, opAssoc.LEFT),
            (NOT, 1, opAssoc.RIGHT),
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
# def evaluate_metrics(llm_outputs: list[str], ground_truths: list[str]):
#     """
#     Evaluates a batch of LLM outputs against their corresponding ground truths.
#     Calculates overall accuracy and false positive rate.
#     """
#     assert len(llm_outputs) == len(ground_truths), "Mismatched list lengths"
#     R = W = 0
#     GT = RS = len(ground_truths)
#     for llm_output, ground_truth in zip(llm_outputs, ground_truths):
#         llm_clean = llm_output.strip().replace(" ", "")
#         gt_clean = ground_truth.strip().replace(" ", "")
#         if llm_clean == gt_clean:
#             R += 1
#         else:
#             W += 1
#     accuracy = R / GT if GT > 0 else 0
#     false_positive = W / RS if RS > 0 else 0
#     return accuracy, false_positive, R, W, GT, RS

def calculate_bleu_score(llm_output: str, ground_truth: str) -> float:
    reference = [ground_truth.split()]
    candidate = llm_output.split()
    return sentence_bleu(reference, candidate)

# --- Define TEST_REQUIREMENTS with Human Gold Standards ---
TEST_REQUIREMENTS = [
    {
        "natural_language": "The alarm must sound only if the window is broken OR the fire detector triggers.",
        "ground_truth": "ALARM_SOUNDS IMPLIES (WINDOW_BROKEN OR FIRE_DETECTOR_TRIGGERS)"
    },
    {
        "natural_language": "If a fire is detected, an alert must activate.",
        "ground_truth": "FIRE_DETECTED IMPLIES ALERT_ACTIVATED"
    },
    {
        "natural_language": "The sprinkler activates only if the fire alarm is triggered.",
        "ground_truth": "SPRINKLER_ACTIVATE IMPLIES FIRE_ALARM_TRIGGERED"
    },
    {
        "natural_language": "If motion is detected and it is night, the lights must turn on.",
        "ground_truth": "(MOTION_DETECTED AND IS_NIGHT) IMPLIES LIGHTS_ON"
    },
    {
        "natural_language": "The door locks if and only if the system is armed.",
        "ground_truth": "DOOR_LOCKED IFF SYSTEM_ARMED"
    },
    {
        "natural_language": "If the temperature exceeds the threshold, the cooling system must activate.",
        "ground_truth": "TEMP_EXCEEDS_THRESHOLD IMPLIES COOLING_SYSTEM_ACTIVATE"
    },
    {
        "natural_language": "Only if a user is authenticated can access be granted.",
        "ground_truth": "ACCESS_GRANTED IMPLIES USER_AUTHENTICATED"
    },
    {
        "natural_language": "If the password is incorrect, access must be denied.",
        "ground_truth": "PASSWORD_INCORRECT IMPLIES ACCESS_DENIED"
    },
    {
        "natural_language": "The emergency lights turn on if power is lost or a fire is detected.",
        "ground_truth": "(POWER_LOST OR FIRE_DETECTED) IMPLIES EMERGENCY_LIGHTS_ON"
    },
    {
        "natural_language": "If the battery is low, a warning must display.",
        "ground_truth": "BATTERY_LOW IMPLIES WARNING_DISPLAYED"
    },
    {
        "natural_language": "A backup server activates only if the main server fails.",
        "ground_truth": "BACKUP_SERVER_ACTIVE IMPLIES MAIN_SERVER_FAILED"
    },
    {
        "natural_language": "If an unauthorized access is detected, the security alert is raised.",
        "ground_truth": "UNAUTHORIZED_ACCESS_DETECTED IMPLIES SECURITY_ALERT_RAISED"
    },
    {
        "natural_language": "Only if maintenance is scheduled, the machine will stop.",
        "ground_truth": "MACHINE_STOPPED IMPLIES MAINTENANCE_SCHEDULED"
    },
    {
        "natural_language": "If the humidity level drops too low, the humidifier must turn on.",
        "ground_truth": "LOW_HUMIDITY_LEVEL IMPLIES HUMIDIFIER_ON"
    },
    {
        "natural_language": "The camera starts recording only if motion is detected.",
        "ground_truth": "CAMERA_RECORDING IMPLIES MOTION_DETECTED"
    },
    {
        "natural_language": "If the window is open and it rains, an alert must be sent.",
        "ground_truth": "(WINDOW_OPEN AND RAINING) IMPLIES ALERT_SENT"
    },
    {
        "natural_language": "The heating system activates only if the temperature is below the set point.",
        "ground_truth": "HEATING_SYSTEM_ACTIVE IMPLIES TEMP_BELOW_SETPOINT"
    },
    {
        "natural_language": "If the server is down, an error log must be generated.",
        "ground_truth": "SERVER_DOWN IMPLIES ERROR_LOG_GENERATED"
    },
    {
        "natural_language": "Only if the user has admin rights can settings be changed.",
        "ground_truth": "SETTINGS_CHANGED IMPLIES USER_HAS_ADMIN_RIGHTS"
    },
    {
        "natural_language": "If the water level is too high, the valve must open.",
        "ground_truth": "WATER_LEVEL_TOO_HIGH IMPLIES VALVE_OPEN"
    },
    {
        "natural_language": "The system must activate if and only if the status is active.",
        "ground_truth": "SYSTEM_ACTIVATED IFF STATUS_ACTIVE"
    }
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
        
        llm_boolean_logic_output = ask_llm_for_boolean_logic(req)
        print(f"LLM-generated Boolean Logic: {llm_boolean_logic_output}")
        
        # Collect outputs for potential batch metrics later
        llm_outputs.append(llm_boolean_logic_output)
        ground_truths.append(ground_truth_req)

        # --- Parse and Process LLM output ---
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
            # print("MEOW")
            print("Transformed tree for 'IMPLIES' (Conceptual):")
            pretty_print_tree(transformed_tree)
            
            print("\n" + "="*50 + "\n")
            
            print("\n--- Processing LLM Output ---")
            
            # 1. Extract variables from LLM's output
            llm_variables = set()
            extract_variables(parsed_tree.asList(), llm_variables)
            print(f"Extracted Variables from LLM: {llm_variables}")
            
            # 2. Get semantic equivalents for each variable
            semantic_map = {}
            for var in llm_variables:
                equivalents = get_semantic_equivalents(var)
                print(f" -> Found equivalents for '{var}': {equivalents}")
                # Store the original variable and its equivalents
                semantic_map[var] = [var] + equivalents
            
            print("\n--- Final Semantic Mapping Dictionary ---")
            print(semantic_map)
            
            # 3. Parse ground truth logic and extract its variables
            parsed_human_tree = parse_boolean_logic(ground_truth_req)
            human_variables = set()
            if parsed_human_tree:
                 extract_variables(parsed_human_tree.asList(), human_variables)
                 print(f"\nExtracted Variables from Human Gold Standard: {human_variables}")
            
            # Now you have llm_variables, human_variables, and the semantic_map.
            # Your next step is to create a single canonical mapping and rewrite both trees
            # before moving to formal consistency checks.
            
            print("\n" + "="*70 + "\n")
        else:
            print("Skipping processing due to parsing error.")
            print("\n" + "="*70 + "\n")
            
    # --- Example of BLEU score calculation on all test cases ---
    # print("\n--- Batch Metrics ---")
    # if llm_outputs and ground_truths:
    #     avg_bleu = sum(calculate_bleu_score(llm, gt) for llm, gt in zip(llm_outputs, ground_truths)) / len(llm_outputs)
    #     print(f"Average BLEU Score across all tests: {avg_bleu:.4f}")
    #     accuracy, false_positive, R, W, GT, RS = evaluate_metrics(llm_outputs, ground_truths)
    #     print(f"Evaluation Metrics:")
    #     print(f"  Accuracy (ACC): {accuracy:.4f} (Right: {R}, Total: {GT})")