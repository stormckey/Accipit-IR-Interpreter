from __future__ import annotations
from lark import Lark, Transformer, ast_utils, Token, UnexpectedInput
from dataclasses import dataclass
from typing import Union, Any
from enum import Enum

import sys
import argparse

class SemanticError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

@dataclass
class IRNode(ast_utils.Ast):
    """
    IRNode is the base class for all IR nodes.
    """

    def __str__(self) -> str:
        raise NotImplementedError("IRNode.__str__() is not implemented.")

@dataclass
class IntConst(IRNode):
    value: int

    def __str__(self):
        return str(self.value)
    
@dataclass
class NoneConst(IRNode):
    name: Token
    
    def __str__(self):
        return "none"
    
@dataclass
class UnitConst(IRNode):
    name: Token
    
    def __str__(self):
        return "()"
    
class Region(Enum):
    GLOBAL = "@"
    PARAM = "#"
    LOCAL = "%"
    
@dataclass 
class Ident:
    region: Region 
    name: str

    def __str__(self):
        return self.region.value + self.name
    
Value = Union[IntConst, NoneConst, UnitConst, Ident]

@dataclass
class I32(IRNode):
    _name: Token
    
    def __str__(self):
        return "i32"

@dataclass
class Unit(IRNode):
    _name: Token
    
    def __str__(self):
        return "()"
    
@dataclass
class Pointer:
    name: str
    
    def __str__(self):
        return self.name

@dataclass
class FunType:
    params: list[Token]
    ret: Token
    
    def __str__(self):
        return f"fn ({', '.join(str(param) for param in self.params)}) -> {self.ret}"
    
Type = Union[I32, Unit, Pointer, FunType]

class Environment():
    def __init__(self):
        self.global_env: dict[str, tuple[Type, Any]] = {}
        self.stack: list[dict[str, tuple[Type, Any]]] = []
        
    def push_stack(self):
        self.stack.append({})

    def pop_stack(self):
        self.stack.pop()

    def add_global(self, name: str, tpe: Type, value: Any):
        if name in self.global_env:
            raise SemanticError(f"Global identifier {name} is defined twice.")
        self.global_env[name] = (tpe, value)
        
    def update_global(self, name: str, value: Any):
        if name not in self.global_env:
            raise SemanticError(f"Global identifier {name} is not defined.")
        self.global_env[name] = (self.global_env[name][0], value)
        
    def add_local(self, name: str, tpe: Type, value: Any):
        if name in self.stack[-1]:
            raise SemanticError(f"Local identifier {name} is defined twice.")
        self.stack[-1][name] = (tpe, value)
        
    def update_local(self, name: str, value: Any):
        if name not in self.stack[-1]:
            raise SemanticError(f"Local identifier {name} is not defined.")
        self.stack[-1][name] = (self.stack[-1][name][0], value)

env = Environment()

@dataclass
class BinExpr(IRNode):
    binop: Token
    v1: Value
    v2: Value
    
    def __str__(self):
        return f"{self.binop} {self.v1}, {self.v2}"
    
class Binop(Enum):
    Add = "add"
    Sub = "sub"
    Mul = "mul"
    Div = "div"
    Rem = "rem"
    And = "and"
    Or = "or"
    Xor = "xor"
    Eq = "eq"
    Ne = "ne"
    Lt = "lt"
    Le = "le"
    Gt = "gt"
    Ge = "ge"
    
def str2binop(s: str) -> Binop:
    return Binop[s.capitalize()]

@dataclass
class Alloca(IRNode):
    tpe: Type
    size: IntConst
    
    def __str__(self):
        return f"alloca {self.tpe}, {self.size}"
    
@dataclass
class Load(IRNode):
    name: Ident
    
    def __str__(self):
        return f"load {self.name}"

@dataclass
class Store(IRNode):
    value: Value
    name: Ident
    
    def __str__(self):
        return f"store {self.value}, {self.name}"

@dataclass
class Gep:
    tpe: Type
    name: Ident
    offsets: list[tuple[int, int]]
    
    def __str__(self):
        indexing = ", ".join(f"{idx} < {dim}" for idx, dim in self.offsets)
        return f"offset {self.tpe}, {self.name}, {indexing}"
    
@dataclass
class Fncall:
    name: Ident
    args: list[Value]

    def __str__(self):
        return f"call {self.name} ({', '.join(str(arg) for arg in self.args)}"
    
ValueBindingOp = Union[BinExpr, Gep, Fncall, Alloca, Load, Store]

@dataclass
class ValueBindingUntyped(IRNode):
    name: Ident
    op: ValueBindingOp
    
    def __str__(self):
        return f"let {self.name} = {self.op}"
    
@dataclass
class ValueBindingTyped(IRNode):
    name: Ident
    type: Type
    op: ValueBindingOp
    
    def __str__(self):
        return f"let {self.name}: {self.type} = {self.op}"
    
ValueBinding = Union[ValueBindingUntyped, ValueBindingTyped]

@dataclass
class Br(IRNode):
    cond: Value
    label1: Ident
    label2: Ident
    
    def __str__(self):
        return f"br {self.cond}, label {self.label1}, label {self.label2}"
    
@dataclass
class Jmp(IRNode):
    label: Ident
    
    def __str__(self):
        return f"jmp label {self.label}"

@dataclass
class Ret(IRNode):
    _keyword: Token
    value: Value
    
    def __str__(self):
        return f"ret {self.value}"
    
Terminator = Union[Br, Jmp, Ret]

@dataclass
class PList:
    params: list[tuple[Ident, Type]]
    
    def __str__(self):
        return ", ".join(f"{name} : {tpe}" for name, tpe in self.params)
    
@dataclass
class BasicBlock:
    label: Ident
    bindings: list[ValueBinding]
    terminator: Terminator
    
    def __str__(self):
        return f"{self.label}:\n" + "\n".join(str(binding) for binding in self.bindings) + "\n{self.terminator}"
    
@dataclass
class Body:
    bbs: list[BasicBlock]
    
    def __str__(self):
        return "{" + "\n".join(str(bb) for bb in self.bbs) + "}"
    
class GlobalDecl:
    name: Ident
    tpe: Type
    size: IntConst
    values: list[Value]
    
    def __init__(self, name: Ident, tpe: Type, size: IntConst, values: list[Value]):
        self.name = name
        self.tpe = tpe
        self.size = size
        self.values = values
        env.add_global(name.__str__(), tpe, self)
        if values and len(values) != size.value:
            raise SemanticError(f"Global array {name} has size {size} but {len(values)} values are provided.")
    
    def __str__(self):
        if self.values:
            return f"{self.name} : {self.tpe}, {self.size} = [{', '.join(str(value) for value in self.values)}]"
        else:
            return f"{self.name} : {self.tpe}, {self.size}"
        
class FunDefn(IRNode):
    name: Ident
    params: PList
    ret: Type
    body: Body
    
    def __init__(self, name: Ident, params: PList, ret: Type, body: Body):
        self.name = name
        self.params = params
        self.ret = ret
        self.body = body
        env.add_global(name.__str__(), FunType([param[1] for param in params.params], ret), self)
    
    def __str__(self):
        return f"fn {self.name} ({self.params}) -> {self.ret} {self.body}"
    
class FunDecl(IRNode):
    name: Ident
    params: PList
    ret: Type
    
    def __init__(self, name: Ident, params: PList, ret: Type):
        self.name = name
        self.params = params
        self.ret = ret
        env.add_global(name.__str__(), FunType([param[1] for param in params.params], ret), None)
    
    def __str__(self):
        return f"fn {self.name} ({self.params}) -> {self.ret};"
    
Decl = Union[GlobalDecl, FunDefn, FunDecl]

@dataclass
class Program:
    decls: list[Decl]
    
    def __str__(self):
        return "\n".join(str(decl) for decl in self.decls)

class BaseTransformer(Transformer):
    # start = lambda _, children: children
    name = lambda _, children: "".join(children)
    
    int_const = lambda _, n: IntConst(int(n[0]))
    none_const = lambda _, n: NoneConst
    
    SIGNED_INT = lambda _, n: int(n)

    function_type = lambda _, items: FunType(items[:-1], items[-1])
    pointer = lambda _, items: Pointer(items[0].__str__() + "*")
    
    global_ident = lambda _, items: Ident(Region.GLOBAL, items[0])
    param_ident = lambda _, items: Ident(Region.PARAM, items[0])
    local_ident = lambda _, items: Ident(Region.LOCAL, items[0])
    
    binop = lambda _, items: str2binop(items[0])
    
    gep = lambda _, items: Gep(items[0], items[1], [(items[i], items[i+1]) for i in range(2, len(items), 2)])
    
    fncall = lambda _, items: Fncall(items[0], items[1:])
    
    plist = lambda _, items: PList([(items[i], items[i+1]) for i in range(0, len(items), 2)])
    
    bb = lambda _, items: BasicBlock(items[0], items[1:-1], items[-1])
    
    body = lambda _, items: Body(items)
    
    global_decl = lambda _, items: GlobalDecl(items[0], items[1], items[2], items[3:])
    
    program = lambda _, items: Program(items)

accipit_grammar = """
    start : program

    name : /[a-zA-Z.-_]/ /[a-zA-Z0-9.-_]/*

    global_ident : "@" (name | INT)
    param_ident : "#" (name | INT)
    local_ident : "%" (name | INT)
    ?ident : global_ident | param_ident | local_ident

    int_const : SIGNED_INT -> int_const
    none_const : /none/ -> none_const
    ?const : int_const
    | none_const
    | /\(\)/ -> unit_const

    ?value : ident | const
    
    type : /i32/ -> i32
    | /\(\)/ -> unit
    | type /\*/ -> pointer
    | "fn" "(" (type ("," type)*)? ")" "->"  type -> function_type
    
    valuebinding_untyped : "let" ident "=" (binexpr | gep | fncall | alloca | load | store)
    valuebinding_typed : "let" ident ":" type "=" (binexpr | gep | fncall | alloca | load | store)
    ?valuebinding : valuebinding_untyped | valuebinding_typed
    ?terminator : br | jmp | ret
    
    ?binexpr : binop value "," value

    binop : /add/ | /sub/ | /mul/ | /div/ | /rem/ | /and/ | /or/ | /xor/ | /eq/ | /ne/ | /lt/ | /le/ | /gt/ | /ge/
    
    alloca : "alloca" type "," int_const
    
    load : "load" ident
    
    store : "store" value "," ident 

    gep : "offset" type "," ident ( "," "[" value "<" (int_const | none_const) "]" )+
    
    fncall : "call" global_ident ("," value)*
    
    br : "br" value "," "label" local_ident "," "label" local_ident
    jmp : "jmp" "label" local_ident
    ret : /(?<!\w)ret(?!\w)/ value
    
    ?plist : (param_ident ":" type ("," param_ident ":" type)*)?
    
    ?label : local_ident ":"
    
    ?bb : label (valuebinding| terminator)*
    
    body : "{" bb* "}"
    
    global_decl : global_ident ":" "region" type "," int_const ("=" "[" value ("," value)* "]")?
    
    fun_defn : "fn" global_ident "(" plist ")" "->" type body
    
    fun_decl : "fn" global_ident "(" plist ")" "->" type ";"
    
    program : (global_decl | fun_defn | fun_decl)*

   %import common.WS
    %import common.CPP_COMMENT
    %import common.C_COMMENT
    %import common.INT
    %import common.SIGNED_INT

    %ignore WS
    %ignore CPP_COMMENT
    %ignore C_COMMENT
"""

this_module = sys.modules[__name__]
accipit_transformer = ast_utils.create_transformer(this_module, BaseTransformer())
parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer)

def parse(file: str) -> Program:
    with open(file) as f:
        text = f.read()
    try:
        parsed_result = parser.parse(text)
        return parsed_result
    except UnexpectedInput as e:
        print(e.get_context(text))
        print(f"Syntax error at position {e.column}: {e}")
        exit(1)
        
def eval(program: Program) -> tuple[int, int]: 
    step = 0
    return (0, step)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Interpreter for Accipit IR")
    arg_parser.add_argument("file", type=str, help="The IR file to interpret.")
    arg_parser.add_argument("-d", "--debug", action="store_true",
                            help="Whether to print debug info.")
    args = arg_parser.parse_args()
    if args.debug:
        print("Debug mode on.")
        DEBUG = True
    program = parse(args.file)
    return_value, step = eval(program)
    # 0 green, else red
    colored_return_value = f"\033[1;32m{return_value}\033[0m" if return_value == 0 else f"\033[1;31m{return_value}\033[0m"
    print(f'Exit with code {colored_return_value} within {step} steps.')
    exit(return_value)