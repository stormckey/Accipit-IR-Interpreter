from __future__ import annotations
from lark import Lark, Transformer, ast_utils, Token, UnexpectedInput
from dataclasses import dataclass
from typing import Union
from enum import Enum

import sys

@dataclass
class IRNode(ast_utils.Ast):
    """
    IRNode is the base class for all IR nodes.
    """

    def __str__(self) -> str:
        raise NotImplementedError("IRNode.__str__() is not implemented.")

@dataclass
class int_const(IRNode):
    value: int

    def __str__(self):
        return str(self.value)
    
@dataclass
class none_const(IRNode):
    name: Token
    
    def __str__(self):
        return "none"
    
@dataclass
class unit_const(IRNode):
    name: Token
    
    def __str__(self):
        return "()"
    
class region(Enum):
    GLOBAL = "@"
    PARAM = "#"
    LOCAL = "%"
    
@dataclass 
class ident:
    region: region 
    name: str

    def __str__(self):
        return self.region.value + self.name
    
value = Union[int_const, none_const, unit_const, ident]

@dataclass
class i32(IRNode):
    _name: Token
    
    def __str__(self):
        return "i32"

@dataclass
class unit(IRNode):
    _name: Token
    
    def __str__(self):
        return "()"
    
@dataclass
class pointer:
    name: str
    
    def __str__(self):
        return self.name

@dataclass
class function_type:
    params: list[Token]
    ret: Token
    
    def __str__(self):
        return f"fn ({', '.join(str(param) for param in self.params)}) -> {self.ret}"
    
tpe = Union[i32, unit, pointer, function_type]

@dataclass
class binexpr(IRNode):
    binop: Token
    v1: value
    v2: value
    
    def __str__(self):
        return f"{self.binop} {self.v1}, {self.v2}"
    
class binop(Enum):
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
    
def str2binop(s: str) -> binop:
    return binop[s.capitalize()]

@dataclass
class alloca(IRNode):
    tpe: tpe
    size: int_const
    
    def __str__(self):
        return f"alloca {self.tpe}, {self.size}"
    
@dataclass
class load(IRNode):
    name: ident
    
    def __str__(self):
        return f"load {self.name}"

@dataclass
class store(IRNode):
    value: value
    name: ident
    
    def __str__(self):
        return f"store {self.value}, {self.name}"

@dataclass
class gep:
    tpe: tpe
    name: ident
    offsets: list[tuple[int, int]]
    
    def __str__(self):
        indexing = ", ".join(f"{idx} < {dim}" for idx, dim in self.offsets)
        return f"offset {self.tpe}, {self.name}, {indexing}"
    
@dataclass
class fncall:
    name: ident
    args: list[value]

    def __str__(self):
        return f"call {self.name} ({', '.join(str(arg) for arg in self.args)}"
    
valuebindingop = Union[binexpr, gep, fncall, alloca, load, store]

@dataclass
class valuebinding_untyped(IRNode):
    name: ident
    op: valuebindingop
    
    def __str__(self):
        return f"let {self.name} = {self.op}"
    
@dataclass
class valuebinding_typed(IRNode):
    name: ident
    type: tpe
    op: valuebindingop
    
    def __str__(self):
        return f"let {self.name}: {self.type} = {self.op}"
    
valuebinding = Union[valuebinding_untyped, valuebinding_typed]

@dataclass
class br(IRNode):
    cond: value
    label1: ident
    label2: ident
    
    def __str__(self):
        return f"br {self.cond}, label {self.label1}, label {self.label2}"
    
@dataclass
class jmp(IRNode):
    label: ident
    
    def __str__(self):
        return f"jmp label {self.label}"

@dataclass
class ret(IRNode):
    _keyword: Token
    value: value
    
    def __str__(self):
        return f"ret {self.value}"
    
terminator = Union[br, jmp, ret]

@dataclass
class plist:
    params: list[tuple[ident, tpe]]
    
    def __str__(self):
        return ", ".join(f"{name} : {tpe}" for name, tpe in self.params)
    
@dataclass
class bb:
    label: ident
    bindings: list[valuebinding]
    terminator: terminator
    
    def __str__(self):
        return f"{self.label}:\n" + "\n".join(str(binding) for binding in self.bindings) + "\n{self.terminator}"
    
@dataclass
class body:
    bbs: list[bb]
    
    def __str__(self):
        return "{" + "\n".join(str(bb) for bb in self.bbs) + "}"
    
@dataclass
class global_decl:
    name: ident
    tpe: tpe
    size: int_const
    values: list[value]
    
    def __str__(self):
        if self.values:
            return f"{self.name} : {self.tpe}, {self.size} = [{', '.join(str(value) for value in self.values)}]"
        else:
            return f"{self.name} : {self.tpe}, {self.size}"
        
@dataclass
class fun_defn(IRNode):
    name: ident
    params: plist
    ret: tpe
    body: body
    
    def __str__(self):
        return f"fn {self.name} ({self.params}) -> {self.ret} {self.body}"
    
@dataclass
class fun_decl(IRNode):
    name: ident
    params: plist
    ret: tpe
    
    def __str__(self):
        return f"fn {self.name} ({self.params}) -> {self.ret};"
    
decl = Union[global_decl, fun_defn, fun_decl]

@dataclass
class program:
    decls: list[decl]
    
    def __str__(self):
        return "\n".join(str(decl) for decl in self.decls)

class BaseTransformer(Transformer):
    # start = lambda _, children: children
    name = lambda _, children: "".join(children)
    
    SIGNED_INT = lambda _, n: int(n)

    function_type = lambda _, items: function_type(items[:-1], items[-1])
    pointer = lambda _, items: pointer(items[0].__str__() + "*")
    
    global_name = lambda _, items: ident(region.GLOBAL, items[0])
    param_name = lambda _, items: ident(region.PARAM, items[0])
    local_name = lambda _, items: ident(region.LOCAL, items[0])
    
    binop = lambda _, items: str2binop(items[0])
    
    gep = lambda _, items: gep(items[0], items[1], [(items[i], items[i+1]) for i in range(2, len(items), 2)])
    
    fncall = lambda _, items: fncall(items[0], items[1:])
    
    plist = lambda _, items: plist([(items[i], items[i+1]) for i in range(0, len(items), 2)])
    
    bb = lambda _, items: bb(items[0], items[1:-1], items[-1])
    
    body = lambda _, items: body(items)
    
    global_decl = lambda _, items: global_decl(items[0], items[1], items[2], items[4:])
    
    program = lambda _, items: program(items)
    # int_const = lambda self, n: int(n[0])

accipit_grammar = """
    start : program

    name : /[a-zA-Z.-_]/ /[a-zA-Z0-9.-_]/*

    ?ident : "@" (name | INT) -> global_name
    | "#" (name | INT) -> param_name
    | "%" (name | INT) -> local_name

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
    
    fncall : "call" ident ("," value)*
    
    br : "br" value "," "label" ident "," "label" ident
    jmp : "jmp" "label" ident
    ret : /(?<!\w)ret(?!\w)/ value
    
    ?plist : (ident ":" type ("," ident ":" type)*)?
    
    ?label : ident ":"
    
    ?bb : label (valuebinding| terminator)*
    
    body : "{" bb* "}"
    
    global_decl : ident ":" "region" type "," int_const ("=" "[" value ("," value)* "]")?
    
    fun_defn : "fn" ident "(" plist ")" "->" type body
    
    fun_decl : "fn" ident "(" plist ")" "->" type ";"
    
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
# parser = Lark(accipit_grammar, parser="lalr")

# Parse and transform the input string
text = """
        // a is a global array with 105 i32 elements.
        // Suppose in SysY it is a multi-dimension array `int a[5][3][7]`
        @a : region i32, 105


        fn @write_global_var(#value: i32) -> () {
        %entry:
            // offset: 3 * 3 * 7 + 2 * 7 + 4 = 81
            let %0 = offset i32, @a, [3 < 5], [2 < 3], [4 < 7]
            let %1 = store #value, %0
            // offset based on previous offsets.
            // offset: 81 + 1 * 3 * 7 + 2 = 104
            let %2 = offset i32, %0, [1 < 5], [0 < 3], [2 < 7]
            let %3 = add #value, 1
            let %4 = store %3, %2
            ret ()
        }

        fn @read_global_var(#dim1: i32, #dim2: i32, #dim3: i32) -> i32 {
        %entry:
            let %0 = offset i32, @a, [#dim1 < 5], [#dim2 < 3], [#dim3 < 7]
            let %1 = load %0
            ret %1
        }

        fn @main() -> () {
        %entry:
            let %0 = call @write_global_var, 10
            let %1 = call @read_global_var, 3, 2, 4
            let %2 = call @read_global_var, 4, 2, 6
            let %3 = call @putint, %1
            let %4 = call @putch, 10
            let %5 = call @putint, %2
            ret ()
        }
"""

try:
    parsed_result = parser.parse(text)
    print(parsed_result)
except UnexpectedInput as e:
    print(e.get_context(text))
    print(f"Syntax error at position {e.column}: {e}")