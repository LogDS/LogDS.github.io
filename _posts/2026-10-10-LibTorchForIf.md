---
layout: post
title: "Deep-Dive Tutorial: Programmatic Control Flow (`prim::If` & `prim::Loop`) in LibTorch JIT C++"
date: 2026-10-01 12:00:00 +0200
categories: [machine-learning, c++]
tags: [libtorch, pytorch, torchscript, jit]
---


When building modular execution frameworks using PyTorch's native C++ API (**LibTorch**), developers often need to construct computational graphs without relying on string-based Python compilation frontends. This requires manually orchestrating the lower-level Intermediate Representation (IR) using the programmatic `torch::jit::Graph` API.

While linear pipelines are straightforward, injecting structured control flow blocks—specifically **`prim::If`** and **`prim::Loop`**—presents stringent architectural constraints. A single type misalignment or a single parameter index drift will crash LibTorch’s underlying execution engine or corrupt your serialized `.pt` binary models.

This tutorial breaks down the technical definitions of structured control flow directly from the PyTorch JIT architecture documentation, translates them into exact C++ implementations, and analyzes the low-level prerequisites required to achieve stable code execution.

---

## 1. Core Architectural Definitions

Unlike unstructured compilers that use control-flow graphs (CFGs) with explicit jump statements or branch blocks, PyTorch JIT uses **Structured Control Flow** implemented via sub-`Block` hierarchies. This structural enforcement simplifies optimization passes (like constant propagation and shape refinement) and guarantees clear Single-Static Assignment (SSA) form dependencies.

### A. The If-Statement (`prim::If`)
From the documentation, a conditional block is modeled as follows:
* **No Block Inputs:** The `Block` boundaries of an if-statement (`block0` for true, `block1` for false) do not accept entry inputs or parameter arguments.
* **Block Outputs as Φ (Phi) Channels:** Values modified across both code tracks must be declared as block outputs via `registerOutput()`.
* **Symmetric Output Matching:** Both branches must return the exact same number of outputs in the exact same type sequence. The containing `prim::If` node then mirrors these internal registers out to the parent graph block (`addOutput()`), performing a role similar to an SSA Φ function.

```text
%y_1, ..., %y_r = prim::If(%condition)
  block0():  # TRUE BRANCH -> Returns r outputs
    ...
    -> (%t_1, ..., %t_r)
  block1():  # FALSE BRANCH -> Returns r outputs
    ...
    -> (%f_1, ..., %f_r)
```

### B. The Loop-Statement (`prim::Loop`)
The `prim::Loop` node handles both `while` and `for` loop abstractions with a unified semantic schema. It relies heavily on **Loop-Carried Dependencies**:

```text
%y_1, ..., %y_r = prim::Loop(%max_trip_count, %initial_condition, %x_1, ..., %x_r)
  block0(%i, %a_1, ..., %a_r):
    ...
    -> (%iter_condition, %b_1, ..., %b_r)
```

The sequence demands strict layout symmetry across four operational points:
1. **Loop Node Inputs:** Receives `%max_trip_count` (the integer execution ceiling), `%initial_condition` (the entry boolean), and the initial values for all state dependencies (`%x_1, ..., %x_r`).
2. **Block Parameter Inputs:** `loop_block->addInput()` expects the JIT loop iteration index (`%i`) as its absolute first argument (Index 0). The following entries map directly to the active loop-carried variables (`%a_1, ..., %a_r`).
3. **Block Termination Outputs:** At the end of the block, `registerOutput()` requires the trailing continuation condition (`%iter_condition`) at Index 0. Passing a constant `false` here immediately signals an early loop termination (**`break`**). This is followed by the mutated values to carry into the next cycle (`%b_1, ..., %b_r`).
4. **Loop Node Outputs:** The external `loop_node->addOutput()` tracking channels must explicitly match the quantity and types of the internal carried state values (`%y_1, ..., %y_r`), skipping the loop tracking parameters.


## 2. Low-Level Requirements to Ensure Code Stability

To build these subgraphs programmatically without crashing PyTorch's `ScriptModuleSerializer` or throwing evaluation exceptions, you must satisfy three low-level requirements:

### I. Operator Overload Signatures: Tensors vs. Scalars
PyTorch IR relies on strict function overloading. For example, `aten::add` has completely different argument counts depending on the input types:
* **Tensor Addition:** Requires **3 arguments**: `(Tensor self, Tensor other, Scalar alpha)`.
* **Primitive Scalar Addition:** Requires **exactly 2 arguments**: `(int self, int other) -> int`. 

When manipulating loop counters or scalar variables, passing a third argument like `alpha` to a purely scalar addition breaks schema validation and corrupts the execution trace.
```cpp
// CORRECT: Primitive scalar addition for loop counters (2 arguments)
torch::jit::Node* next_counter = graph->create(torch::jit::aten::add, {counter_curr, const_1});
```

### II. Output Layer Purging via `eraseOutput`
When creating high-level conditional nodes like `prim::If` or `prim::Loop` using `graph->create`, LibTorch may automatically assign implicit fallback slots or track floating output nodes in memory. When explicitly designing structured Φ registers or type definitions, you must clear out accidental background slots using a cleanup loop before running your type configuration:
```cpp
while (if_node->outputs().size() > 0) {
    if_node->eraseOutput(0);
}
```
Failing to purge these channels leads to asymmetries and causes the Python printer to output extra ghost variables (like `_10` or `_24`), crashing your code reload step.

### III. Explicit Class Allocation for Serializer Contexts
If a graph is intended to be exported and saved to a file via `module.save()`, any custom data configurations like `TupleType` structures can be created anonymously. The `ScriptModuleSerializer` traverses the complete compilation database to construct clean Python environment representations. 

```cpp
auto tuple_return_type = c10::TupleType::create({int_type, tensor_type});
cu->register_type(tuple_return_type); // CRITICAL: Makes the type discoverable by the serializer
```

## C++ LibTorch JIT

I now showcase the code required to generate the Python code stated above:

```cpp
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <memory>
#include <limits>

int main(void) {
auto cu = std::make_shared<torch::jit::CompilationUnit>();
    auto class_name = c10::QualifiedName("__torch__.ConditionalLoopModule");
    auto class_type = c10::ClassType::create(class_name, cu, /*is_module=*/true);
    torch::jit::Module module_(cu, class_type);

    auto graph = std::make_shared<torch::jit::Graph>();

    // 1. Module input for the forward function, represented as a dataflow graph
    torch::jit::Value* self_val = graph->addInput("self")->setType(class_type);
    torch::jit::Value* input_val = graph->addInput("input_val")->setType(c10::TensorType::get());

    // =========================================================================
    // 2. Main constants 
    // =========================================================================
    torch::jit::Value* const_max_int = graph->insertConstant(std::numeric_limits<int64_t>::max())->setType(c10::IntType::get()); // %80
    torch::jit::Value* const_true    = graph->insertConstant(true)->setType(c10::BoolType::get());                     // %8
    torch::jit::Value* status_init   = graph->insertConstant((int64_t)0)->setType(c10::IntType::get());                // %status.1
    torch::jit::Value* const_2       = graph->insertConstant((int64_t)2)->setType(c10::IntType::get());                // %3
    torch::jit::Value* const_20      = graph->insertConstant((int64_t)20)->setType(c10::IntType::get());               // %5
    torch::jit::Value* const_1       = graph->insertConstant((int64_t)1)->setType(c10::IntType::get());                // %22
    torch::jit::Value* loop_counter_init = graph->insertConstant((int64_t)0)->setType(c10::IntType::get());            // Contatore iniziale range(20)
    torch::jit::Value* const_false   = graph->insertConstant(false)->setType(c10::BoolType::get());                   // %55

    // Some float constants
    torch::jit::Value* const_0_05    = graph->insertConstant(0.05)->setType(c10::FloatType::get());                    // %13
    torch::jit::Value* const_0_01    = graph->insertConstant(0.01)->setType(c10::FloatType::get());                    // %17
    torch::jit::Value* const_10      = graph->insertConstant(10.0)->setType(c10::FloatType::get());                    // %25

    // 3. Initial computation, outside the loop value = input_val * 2
    torch::jit::Node* mul_node = graph->create(torch::jit::aten::mul, {input_val, const_2});
    graph->insertNode(mul_node);
    torch::jit::Value* value_init = mul_node->output()->setType(c10::TensorType::get());

    // =========================================================================
    // 4. CREAZIONE DEL NODO PRIM::LOOP
    // So to slavishly follow PyTorch JIT requirements:
    // input order: max_trip, cond_in, status_init, value_init, counter_init (status_init is reused as counter within the target graph)
    // =========================================================================
    torch::jit::Node* loop_node = graph->create(torch::jit::prim::Loop, {const_max_int, const_true, status_init, value_init, status_init});
    graph->insertNode(loop_node);

    torch::jit::Block* loop_block = loop_node->addBlock();

    // Input of the native IR block
    loop_block->addInput()->setType(c10::IntType::get()); // Implicit JIT Iterator
    torch::jit::Value* status_curr = loop_block->addInput()->setType(c10::IntType::get());   // Int
    torch::jit::Value* value_curr  = loop_block->addInput()->setType(c10::TensorType::get());  // Tensor
    torch::jit::Value* counter_curr = loop_block->addInput()->setType(c10::IntType::get()); // Int (the counter

    // --- Some Internal Loop Logic ---
    // value = value * (i+1) * 0.05
    // Incrementing the counter for the for loop
    torch::jit::Node* next_counter = graph->create(torch::jit::aten::add, {counter_curr, const_1});
    loop_block->appendNode(next_counter);
    torch::jit::Value* counter = next_counter->output()->setType(c10::IntType::get());
    torch::jit::Node* mul_i = graph->create(torch::jit::aten::mul, {value_curr, counter});
    loop_block->appendNode(mul_i);
    torch::jit::Node* mul_0_05 = graph->create(torch::jit::aten::mul, {mul_i->output(), const_0_05});
    loop_block->appendNode(mul_0_05);
    torch::jit::Value* value_scaled = mul_0_05->output()->setType(c10::TensorType::get());

    // Condition: torch::all(value > 0.01)
    torch::jit::Node* gt_node = graph->create(torch::jit::aten::gt, {value_scaled, const_0_01});
    loop_block->appendNode(gt_node);
    torch::jit::Node* all_node = graph->create(torch::jit::aten::all, {gt_node->output()});
    loop_block->appendNode(all_node);
    torch::jit::Node* bool_cast = graph->create(torch::jit::aten::Bool, {all_node->output()});
    loop_block->appendNode(bool_cast);
    torch::jit::Value* if_condition = bool_cast->output()->setType(c10::BoolType::get());

    // =========================================================================
    // 5. CREATING THE CONDITIONAL PRIM::IF WITHIN THE LOOP
    // =========================================================================
    torch::jit::Node* if_node = graph->create(torch::jit::prim::If, {if_condition});
    loop_block->appendNode(if_node);

    torch::jit::Block* then_block = if_node->addBlock();
    torch::jit::Block* else_block = if_node->addBlock();

    //  THEN (True) branch -> output: (continue, status, value)
    torch::jit::Node* then_add = graph->create(torch::jit::aten::add, {value_scaled, const_10, const_1});
    then_block->appendNode(then_add);
    torch::jit::Value* result = then_add->output()->setType(c10::TensorType::get());

    then_block->registerOutput(const_true);
    then_block->registerOutput(const_1);
    then_block->registerOutput(result);

    //  ELSE (False / BREAK) branch -> output: (continue, status, value)
    else_block->registerOutput(const_false);
    else_block->registerOutput(status_curr);
    else_block->registerOutput(value_scaled);

    // -------------------------------------------------------------------------
    // CRITICAL FIX: Clearing and forcing the final 3 outputs over nodo prim::If
    // This ensures to completely remove a host fourth registry (_10) from our final Python code!
    // -------------------------------------------------------------------------
    while (if_node->outputs().size() > 0) {
        if_node->eraseOutput(0);
    }

    torch::jit::Value* if_continue = if_node->addOutput()->setType(c10::BoolType::get());
    torch::jit::Value* if_status   = if_node->addOutput()->setType(c10::IntType::get());
    torch::jit::Value* if_value    = if_node->addOutput()->setType(c10::TensorType::get());
    // -------------------------------------------------------------------------


    // Check maximum bounded loop iterations: counter < 20
    // Simulating the for loop with a while-condition over the cycle upper bound
    torch::jit::Node* lt_node = graph->create(torch::jit::aten::lt, {counter, const_20});
    loop_block->appendNode(lt_node);
    torch::jit::Value* ltout = lt_node->output()->setType(c10::BoolType::get());

    // Total stopping condition: (counter < 20) AND (if_continue)
    torch::jit::Node* and_node = graph->create(torch::jit::aten::__and__, {ltout, if_continue});
    loop_block->appendNode(and_node);
    torch::jit::Value* loop_keep_going = and_node->output()->setType(c10::BoolType::get());


    // =========================================================================
    // 6. Setting the total output registries
    // Index 0: Continuation gate signal (loop_keep_going)
    // Index 1+: Loops back modified current tracking states (status, value, counter)
    // =========================================================================
    loop_block->registerOutput(loop_keep_going);
    loop_block->registerOutput(if_status);
    loop_block->registerOutput(if_value);
    loop_block->registerOutput(counter);

    // =========================================================================
    // 7. Setting outer block's results
    // Clean, trace, and align external return values with inner dependencies 1:1
    // =========================================================================
    while (loop_node->outputs().size() > 0) {
        loop_node->eraseOutput(0);
    }
    torch::jit::Value* final_status  = loop_node->addOutput()->setType(c10::IntType::get());
    torch::jit::Value* final_value   = loop_node->addOutput()->setType(c10::TensorType::get());
    torch::jit::Value* final_counter = loop_node->addOutput()->setType(c10::IntType::get());

    // =========================================================================
    // 8. Returning multiple values as elements of a tuple
    // Pack only final_status and final_value, dropping the trailing loop counter variable
    // =========================================================================
    torch::jit::Node* tuple_node = graph->create(torch::jit::prim::TupleConstruct, {final_status, final_value});
    graph->insertNode(tuple_node);
    graph->registerOutput(tuple_node->output());

    auto tensor_type = c10::TensorType::get();
    auto int_type = c10::IntType::get();
    auto tuple_name = c10::QualifiedName("__torch__.MyReturnTypeTuple");
    auto tuple_return_type = c10::TupleType::create({int_type, tensor_type});
    cu->register_type(tuple_return_type);
    tuple_node->output()->setType(tuple_return_type);

    // =========================================================================
    // 9. Final schema compilation
    // =========================================================================
    std::vector<c10::Argument> arguments = {
        c10::Argument("self", class_type),
        c10::Argument("input_val", tensor_type)
    };
    std::vector<c10::Argument> returns = {
        c10::Argument("", tuple_return_type)
    };

    c10::QualifiedName method_name(class_name, "forward");
    c10::FunctionSchema schema(method_name.name(), "", arguments, returns);

    auto method = module_._ivalue()->compilation_unit()->create_function(method_name, graph);
    method->setSchema(std::move(schema));
    class_type->addMethod(method);

    module_.save("conditional_subgraph.pt");
    std::cout << "Modello salvato." << std::endl;

    return 0;
    }
```

## Python driver 

The following execution script handles loading your generated graph, executes a step-by-step mathematical state validation tracer, and inspects the structural text layout.


```python
import torch


def simulate_evolution(input_val, max_cycles=20):
    """
    Simula esattamente il comportamento del grafo JIT C++ ciclo per ciclo
    per mostrare l'evoluzione delle variabili e calcolare il risultato atteso.
    """
    print("\n--- Pure Python simulation of the expected behaviour of our JIT code ---")
    status = 0
    # outer computation value = input_val * 2
    value = input_val.clone() * 2
    print(f"Initial configuration -> status: {status}, value:\n{value}")

    for i in range(max_cycles):
        # 1. Multiplication per the next iteration index
        value_scaled = value * (i+1) * 0.05

        # 2. Checking the condition over the scaled value
        condition = torch.all(value_scaled > 0.01).item()

        print(f"\n[Iteration #{i:02d}] -> Times i={i} and 0.05")
        print(f"         -> After multiplication:\n{value_scaled}")
        print(f"         -> Check (value > 0.01)? {condition}")

        if condition:
            status = 1
            value = value_scaled + 10.0
            print(f"         -> [THEN BRANCH] UPDATE! status: {status}, value (con +10):\n{value}")
        else:
            print(f"         -> [ELSE BRANCH] !!! BREAK TRIGGERED !!! We stop here.")
            # Al break, il valore finale che sopravvive è value_scaled (come da blocco else_block in C++)
            value = value_scaled
            break

    return status, value


def test_model_with_evolution(model_path="conditional_subgraph.pt"):
    print(f"--- Loading the JIT model from: {model_path} ---")
    try:
        model = torch.jit.load(model_path)
    except Exception as e:
        print(f"Error while loading the .pt file: {e}")
        return

    # Pick an input that is not going to zero
    input_tensor = torch.ones(2, 2) * 5.0

    # 1. Performing the Python code simulation
    expected_status, expected_value = simulate_evolution(input_tensor, max_cycles=20)

    # 2. Running the JIT-compiled code
    print("\n--- JIT running ---")
    jit_status, jit_value = model(input_tensor)

    print(f"\n=== Checking the final results ===")
    print(f" JIT Status: {jit_status} | Expected status: {expected_status}")
    print(f" JIT Value:\n{jit_value}")
    print(f"Expected outcome:\n{expected_value}")

    # Controllo di uguaglianza formale
    outputs_match = torch.equal(jit_value, expected_value) and (jit_status == expected_status)
    print(f"\nWas the computation correct? {'SÌ ✅' if outputs_match else 'NO ❌'}")


if __name__ == "__main__":
    test_model_with_evolution()

```

By systematically purging intermediate nodes using `eraseOutput` and maintaining proper scalar signature definitions, your low-level LibTorch components will translate flawlessly back into cross-compatible Python execution contexts.

